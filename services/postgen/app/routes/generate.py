import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import httpx

from shared.db.mongo_db import get_database, POSTS_COLLECTION_NAME
from app.llm import generate_hybrid_post, generate_simple_post
from app.utils.hybrid_retriever import create_retriever
from app.config import GENERATED_POSTS_COLLECTION_NAME, SCRAPER_SERVICE_URL
from app.utils.pinecone import pinecone_service
from app.models.generated_post import GeneratedPostItem

logger = logging.getLogger(__name__)
router = APIRouter()


class GeneratePostRequest(BaseModel):
    prompt: str = Field(..., description="The main prompt for post generation")
    topic: Optional[str] = Field(None, description="Topic for the post")
    tone: Optional[str] = Field(None, description="Tone of the post")
    length: Optional[str] = Field(None, description="Desired length of the post")
    audience: Optional[str] = Field(None, description="Target audience")
    hashtag: Optional[str] = Field(None, description="Hashtag to get trending samples")
    num_variations: Optional[int] = Field(1, ge=1, le=3, description="Number of variations (1-3)")
    username: Optional[str] = Field(None, description="LinkedIn username")


async def ensure_user_posts_in_pinecone(username: str, db) -> bool:
    try:
        query_response = pinecone_service.index.query(
            vector=[0.0] * 1536,
            filter={"username": username},
            top_k=1,
            include_metadata=True
        )
        if query_response.matches:
            return True

        # Fetch from MongoDB if not in Pinecone
        user_posts_cursor = db[POSTS_COLLECTION_NAME].find({"username": username})
        user_document = await user_posts_cursor.to_list(length=1)
        if user_document:
            all_posts = user_document[0].get("posts", [])
            recent_posts = sorted(all_posts, key=lambda x: x.get("scraped_at", ""), reverse=True)[:10]
            if recent_posts:
                success = pinecone_service.store_user_posts(username, recent_posts)
                if success:
                    return True
        return False
    except Exception as e:
        logger.error(f"Error ensuring user posts in Pinecone: {e}")
        return False


async def fetch_trending_posts(hashtag: str) -> str:
    if not hashtag:
        return ""
    
    try:
        timeout = httpx.Timeout(120.0, connect=15.0)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                f"{SCRAPER_SERVICE_URL}/scraper/hashtag/posts",
                params={"hashtag": hashtag, "n_posts": 5}
            )
            
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    posts = data.get("posts", [])
                    if posts:
                        trending_text = "\n\n".join([
                            f"Trending post {i+1}: {post.get('text', '')[:200]}..." 
                            for i, post in enumerate(posts[:3]) 
                            if post.get('text', '').strip()
                        ])
                        
                        if trending_text:
                            return trending_text
                except Exception:
                    return ""
            
            return ""
    
    except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout):
        logger.warning(f"Timeout fetching trending posts for hashtag '{hashtag}'")
        return ""
    except Exception as e:
        logger.error(f"Error fetching trending posts for hashtag '{hashtag}': {repr(e)}")
        return ""


async def save_generated_posts(db, username: str, post_items: List[GeneratedPostItem]) -> str:
    collection = db[GENERATED_POSTS_COLLECTION_NAME]
    existing_doc = await collection.find_one({"username": username})
    
    try:
        if existing_doc:
            await collection.update_one(
                {"username": username},
                {
                    "$push": {"generated_posts": {"$each": [item.dict() for item in post_items]}},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            return str(existing_doc["_id"])
        else:
            new_doc = {
                "username": username,
                "user_id": None,
                "generated_posts": [item.dict() for item in post_items],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            result = await collection.insert_one(new_doc)
            return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error saving generated posts: {e}")
        raise


@router.post("/generate")
async def generate_post(req: GeneratePostRequest, db=Depends(get_database)):
    try:
        username = req.username
        if not username:
            raise HTTPException(status_code=400, detail="Username is required")
        
        # Clean LinkedIn URL if provided
        if 'linkedin.com' in username and '/in/' in username:
            username = username.split('/in/')[-1].rstrip('/')

        # Get user context and trending posts
        posts_available = await ensure_user_posts_in_pinecone(username, db)
        trending_posts = await fetch_trending_posts(req.hashtag)

        # Generate posts based on available context
        if posts_available:
            retriever = create_retriever(alpha=0.5, top_k=5, username=username)
            generated_posts = generate_hybrid_post(
                prompt=req.prompt,
                retriever=retriever,
                trending_posts=trending_posts,
                topic=req.topic,
                tone=req.tone,
                length=req.length,
                audience=req.audience,
                num_variations=req.num_variations or 1
            )
            generation_method = "hybrid_rag_trending"
        else:
            enhanced_prompt = req.prompt
            if trending_posts:
                enhanced_prompt += f"\n\nContext - Current trending posts:\n{trending_posts}\n\nCreate an original post inspired by these trends."
            generated_posts = generate_simple_post(prompt=enhanced_prompt, num_variations=req.num_variations or 1)
            generation_method = "simple_with_trending" if trending_posts else "simple_only"

        # Prepare post items for database
        post_items = [
            GeneratedPostItem(
                original_prompt=req.prompt,
                generated_text=post_text,
                parameters={
                    "topic": req.topic,
                    "tone": req.tone,
                    "length": req.length,
                    "audience": req.audience,
                    "hashtag": req.hashtag,
                    "generation_method": generation_method
                },
                style_sample_used=posts_available,
                trending_sample_used=bool(trending_posts),
                variation_number=i + 1,
                created_at=datetime.utcnow()
            )
            for i, post_text in enumerate(generated_posts)
        ]

        doc_id = await save_generated_posts(db, username, post_items)

        return {
            "success": True,
            "variations": generated_posts,
            "username_used": username,
            "generation_method": generation_method,
            "user_context_available": posts_available,
            "trending_context_available": bool(trending_posts),
            "saved_to_db": True,
            "document_id": doc_id,
            "total_variations": len(generated_posts)
        }

    except Exception as e:
        logger.error(f"Error in generate_post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{username}")
async def get_user_history(username: str, limit: int = 20, db=Depends(get_database)):
    try:
        collection = db[GENERATED_POSTS_COLLECTION_NAME]
        user_doc = await collection.find_one({"username": username})
        if not user_doc:
            return {"success": True, "posts": [], "total_posts": 0}
        
        all_posts = user_doc.get("generated_posts", [])
        sorted_posts = sorted(all_posts, key=lambda x: x.get("created_at"), reverse=True)
        return {
            "success": True, 
            "posts": sorted_posts[:limit], 
            "total_posts": len(all_posts)
        }
    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        raise HTTPException(status_code=500, detail=str(e))