# routes/scraping.py - Enhanced version with better error handling and debugging
from fastapi import APIRouter, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Literal
from datetime import datetime, timedelta, timezone
import os
import json
import traceback
from pathlib import Path
from shared.db.mongo_db import get_database
from app.models.post import PostInDB, UserPosts, HashtagPosts
from app.utils.linkedin_bot import LinkedInBot

router = APIRouter()

# Directories for saving posts
BASE_DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR = BASE_DATA_DIR / "profile_posts"
HASHTAG_DATA_DIR = BASE_DATA_DIR / "hashtag_posts"

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(HASHTAG_DATA_DIR).mkdir(parents=True, exist_ok=True)

# ======== Schemas ========
Source = Literal["profile", "hashtag"]

class PostData(BaseModel):
    text: str = ""
    likes: int = 0
    comments: int = 0
    reposts: int = 0
    engagement: int = 0
    scraped_at: str
    source: Source
    profile_url: Optional[str] = None
    hashtag: Optional[str] = None

class ProfilePostsResponse(BaseModel):
    success: bool
    message: str
    total_posts: int
    posts: List[PostData]
    from_cache: bool
    execution_time_seconds: float
    debug_info: Optional[dict] = None

class HashtagPostsResponse(BaseModel):
    success: bool
    message: str
    total_posts: int
    posts: List[PostData]
    from_cache: bool
    execution_time_seconds: float
    debug_info: Optional[dict] = None

# ======== Helper Functions ========

def extract_profile_identifier(profile_url: str) -> str:
    """Extract username from LinkedIn profile URL"""
    clean_url = profile_url.rstrip('/')
    if '/in/' in clean_url:
        return clean_url.split('/in/')[-1].split('/')[0]
    elif '/company/' in clean_url:
        return clean_url.split('/company/')[-1].split('/')[0]
    else:
        return clean_url.split('/')[-1]

def convert_post_to_response_format(post_dict: dict) -> PostData:
    """Convert database post dict to API response format"""
    return PostData(
        text=post_dict.get("text", ""),
        likes=post_dict.get("likes", 0),
        comments=post_dict.get("comments", 0),
        reposts=post_dict.get("reposts", 0),
        engagement=post_dict.get("engagement", 0),
        scraped_at=post_dict.get("scraped_at", ""),
        source=post_dict.get("source", "profile"),
        profile_url=post_dict.get("profile_url"),
        hashtag=post_dict.get("hashtag")
    )

async def save_profile_posts_to_db(posts_data: List[dict], profile_url: str, db: AsyncIOMotorDatabase) -> List[dict]:
    """Save profile posts to database with proper formatting"""
    username = extract_profile_identifier(profile_url)
    current_time = datetime.utcnow().isoformat()
    formatted_posts = []
    
    for post in posts_data:
        formatted_posts.append({
            "text": post.get("text", ""),
            "likes": post.get("likes", 0),
            "comments": post.get("comments", 0),
            "reposts": post.get("reposts", 0),
            "engagement": post.get("engagement", 0),
            "scraped_at": post.get("scraped_at", current_time),
            "source": "profile",
            "hashtag": None
        })
    
    try:
        existing_doc = await db.posts.find_one({"username": username})
        if existing_doc:
            # Replace existing posts instead of appending to avoid duplicates
            await db.posts.update_one(
                {"username": username},
                {"$set": {
                    "posts": formatted_posts,
                    "profile_url": profile_url.rstrip("/"), 
                    "updated_at": current_time
                }}
            )
        else:
            new_doc = {
                "username": username,
                "profile_url": profile_url.rstrip("/"),
                "posts": formatted_posts,
                "created_at": current_time,
                "updated_at": current_time
            }
            await db.posts.insert_one(new_doc)
        
        print(f"💾 Saved {len(formatted_posts)} posts to database for {username}")
        return formatted_posts
        
    except Exception as e:
        print(f"❌ Save error: {e}")
        traceback.print_exc()
        return []

async def save_hashtag_posts_to_db(posts_data: List[dict], hashtag: str, db: AsyncIOMotorDatabase) -> List[dict]:
    """Save hashtag posts to database"""
    current_time = datetime.utcnow().isoformat()
    formatted_posts = []
    
    for post in posts_data:
        formatted_posts.append({
            "text": post.get("text", ""),
            "likes": post.get("likes", 0),
            "comments": post.get("comments", 0),
            "reposts": post.get("reposts", 0),
            "engagement": post.get("engagement", 0),
            "scraped_at": post.get("scraped_at", current_time),
            "source": "hashtag",
            "profile_url": None,
            "hashtag": hashtag
        })
    
    try:
        existing_doc = await db.hashtag_posts.find_one({"hashtag": hashtag})
        if existing_doc:
            await db.hashtag_posts.update_one(
                {"hashtag": hashtag},
                {"$set": {
                    "posts": formatted_posts,
                    "updated_at": current_time
                }}
            )
        else:
            new_doc = {
                "hashtag": hashtag,
                "posts": formatted_posts,
                "created_at": current_time,
                "updated_at": current_time
            }
            await db.hashtag_posts.insert_one(new_doc)
        
        print(f"💾 Saved {len(formatted_posts)} hashtag posts to database for #{hashtag}")
        return formatted_posts
        
    except Exception as e:
        print(f"❌ Save error: {e}")
        traceback.print_exc()
        return []

def save_posts_to_json(posts_data: List[dict], identifier: str, is_hashtag: bool = False) -> str:
    """Save posts to JSON file for backup and analysis"""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f"{'hashtag' if is_hashtag else 'profile'}_{identifier}_{timestamp}.json"
    filepath = (HASHTAG_DATA_DIR if is_hashtag else DATA_DIR) / filename
    
    # Ensure datetime objects are serializable
    for post in posts_data:
        for key in ["scraped_at", "created_at"]:
            if key in post and isinstance(post[key], datetime):
                post[key] = post[key].isoformat()
    
    json_data = {
        "identifier": identifier,
        "type": "hashtag" if is_hashtag else "profile",
        "scraped_at": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
        "total_posts": len(posts_data),
        "posts": posts_data
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved posts to JSON: {filepath}")
    return str(filepath)

def is_cache_fresh(updated_at: str, max_age_hours: int = 24) -> bool:
    """Check if cached data is still fresh"""
    try:
        updated_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        age = (now - updated_time).total_seconds() / 3600
        return age < max_age_hours
    except:
        return False

# ======== Helper for safe bot close ========
async def _safe_close_bot(bot: Optional[LinkedInBot]):
    """Attempt to close bot and swallow any exceptions (for cleanup)."""
    if not bot:
        return
    try:
        await bot.close()
    except Exception as e:
        # Log but do not raise from cleanup
        print(f"⚠️ Error while closing bot: {e}")
        traceback.print_exc()

# ======== Routes ========

@router.get("/profile/posts", response_model=ProfilePostsResponse)
async def get_user_profile_posts(
    profile_url: str,
    n_posts: int = Query(default=10, ge=1, le=50),
    force_fresh: bool = Query(default=False, description="Force fresh scraping instead of using cache"),
    debug: bool = Query(default=False, description="Enable debug mode for detailed logging"),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get posts from a LinkedIn profile"""
    start_time = datetime.utcnow()
    debug_info = {}
    
    bot = None
    try:
        username = extract_profile_identifier(profile_url)
        debug_info["username"] = username
        debug_info["requested_posts"] = n_posts
        debug_info["profile_url"] = profile_url
        
        print(f"🔍 Fetching posts for profile: {profile_url}")
        print(f"📊 Username extracted: {username}")
        
        # Check cache first (unless force_fresh is True)
        if not force_fresh:
            user_doc = await db.posts.find_one({"username": username})
            if user_doc and "posts" in user_doc:
                cached_posts = user_doc["posts"]
                is_fresh = is_cache_fresh(user_doc.get("updated_at", ""), max_age_hours=24)
                
                debug_info["cached_posts_count"] = len(cached_posts)
                debug_info["cache_is_fresh"] = is_fresh
                debug_info["cache_updated_at"] = user_doc.get("updated_at")
                
                if len(cached_posts) >= n_posts and is_fresh:
                    # Return cached data
                    posts_sorted = sorted(cached_posts, key=lambda x: x.get("scraped_at", ""), reverse=True)[:n_posts]
                    response_posts = [convert_post_to_response_format(post) for post in posts_sorted]
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    print(f"✅ Returning {len(response_posts)} cached posts")
                    
                    return ProfilePostsResponse(
                        success=True, 
                        message=f"Retrieved {len(response_posts)} posts from cache",
                        total_posts=len(response_posts), 
                        posts=response_posts,
                        from_cache=True, 
                        execution_time_seconds=execution_time,
                        debug_info=debug_info if debug else None
                    )
                else:
                    print(f"⚠️ Cache insufficient or stale. Cached: {len(cached_posts)}, Needed: {n_posts}, Fresh: {is_fresh}")
        
        # Scrape fresh data
        print("🤖 Starting LinkedIn bot for fresh scraping...")
        bot = LinkedInBot(
            email=os.getenv("LINKEDIN_EMAIL"), 
            password=os.getenv("LINKEDIN_PASSWORD"), 
            headless=True,
            debug=debug
        )
        
        debug_info["bot_config"] = {
            "headless": True,
            "debug": debug,
            "email_configured": bool(os.getenv("LINKEDIN_EMAIL"))
        }
        
        await bot.start()
        print("🔑 Logging into LinkedIn...")
        await bot.login()
        
        print(f"🔍 Scraping {n_posts} posts from {profile_url}")
        scraped_posts = await bot.scrape_user_posts(profile_url, n_posts)
        
        debug_info["scraped_posts_count"] = len(scraped_posts)
        debug_info["scraping_successful"] = len(scraped_posts) > 0
        
        # CRITICAL: close the bot before doing DB operations or returning response
        await _safe_close_bot(bot)
        bot = None
        
        if not scraped_posts:
            raise Exception("No posts were scraped from the profile")
        
        # Save to database
        print("💾 Saving posts to database...")
        saved_posts = await save_profile_posts_to_db(scraped_posts, profile_url, db)
        
        # Save to JSON backup
        json_file = save_posts_to_json(scraped_posts, username)
        debug_info["json_backup"] = json_file
        
        # Prepare response
        response_posts = [convert_post_to_response_format(post) for post in saved_posts]
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        print(f"✅ Successfully scraped and saved {len(response_posts)} posts")
        
        return ProfilePostsResponse(
            success=True, 
            message=f"Successfully scraped {len(response_posts)} posts from profile",
            total_posts=len(response_posts), 
            posts=response_posts,
            from_cache=False, 
            execution_time_seconds=execution_time,
            debug_info=debug_info if debug else None
        )
        
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        error_message = str(e)
        
        print(f"❌ Error in profile posts endpoint: {error_message}")
        if debug:
            traceback.print_exc()
            debug_info["error_details"] = traceback.format_exc()
        
        # Ensure bot is closed even on error
        await _safe_close_bot(bot)
        
        return ProfilePostsResponse(
            success=False, 
            message=f"Error scraping profile posts: {error_message}", 
            total_posts=0, 
            posts=[], 
            from_cache=False, 
            execution_time_seconds=execution_time,
            debug_info=debug_info if debug else None
        )

@router.get("/hashtag/posts", response_model=HashtagPostsResponse)
async def get_hashtag_posts(
    hashtag: str,
    n_posts: int = Query(default=5, ge=2, le=20),
    force_fresh: bool = Query(default=False, description="Force fresh scraping instead of using cache"),
    debug: bool = Query(default=False, description="Enable debug mode for detailed logging"),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get top posts from a LinkedIn hashtag"""
    start_time = datetime.utcnow()
    debug_info = {}
    
    bot = None
    try:
        clean_hashtag = hashtag.lstrip('#')
        debug_info["hashtag"] = clean_hashtag
        debug_info["requested_posts"] = n_posts
        
        print(f"🏷️ Fetching posts for hashtag: #{clean_hashtag}")
        
        # Check cache first (unless force_fresh is True)
        if not force_fresh:
            hashtag_doc = await db.hashtag_posts.find_one({"hashtag": clean_hashtag})
            
            if hashtag_doc and "posts" in hashtag_doc:
                cached_posts = hashtag_doc["posts"]
                is_fresh = is_cache_fresh(hashtag_doc.get("updated_at", ""), max_age_hours=12)  # Shorter cache for hashtags
                
                debug_info["cached_posts_count"] = len(cached_posts)
                debug_info["cache_is_fresh"] = is_fresh
                debug_info["cache_updated_at"] = hashtag_doc.get("updated_at")
                
                if len(cached_posts) >= 2 and is_fresh:  # Always return top 2
                    posts_sorted_by_engagement = sorted(
                        cached_posts, 
                        key=lambda x: x.get("engagement", 0), 
                        reverse=True
                    )[:2]
                    
                    response_posts = [convert_post_to_response_format(post) for post in posts_sorted_by_engagement]
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    print(f"✅ Returning {len(response_posts)} cached hashtag posts")
                    
                    return HashtagPostsResponse(
                        success=True, 
                        message=f"Retrieved top {len(response_posts)} posts by engagement from cache for #{clean_hashtag}",
                        total_posts=len(response_posts), 
                        posts=response_posts,
                        from_cache=True, 
                        execution_time_seconds=execution_time,
                        debug_info=debug_info if debug else None
                    )
                else:
                    print(f"⚠️ Cache insufficient or stale for hashtag. Cached: {len(cached_posts) if cached_posts else 0}, Fresh: {is_fresh}")
        
        # Scrape fresh data
        print("🤖 Starting LinkedIn bot for hashtag scraping...")
        bot = LinkedInBot(
            email=os.getenv("LINKEDIN_EMAIL"), 
            password=os.getenv("LINKEDIN_PASSWORD"), 
            headless=True,
            debug=debug
        )
        
        debug_info["bot_config"] = {
            "headless": True,
            "debug": debug,
            "email_configured": bool(os.getenv("LINKEDIN_EMAIL"))
        }
        
        await bot.start()
        await bot.login()
        
        # Scrape more posts for better trend analysis
        scrape_count = max(n_posts, 5)  # Scrape at least 5 for good sample
        print(f"🔍 Scraping {scrape_count} posts for hashtag trend analysis...")
        scraped_posts = await bot.scrape_hashtag_posts(clean_hashtag, scrape_count)
        
        debug_info["scraped_posts_count"] = len(scraped_posts)
        debug_info["scraping_successful"] = len(scraped_posts) > 0
        
        # CRITICAL: close the bot before DB write
        await _safe_close_bot(bot)
        bot = None
        
        if not scraped_posts:
            raise Exception(f"No posts were scraped for hashtag #{clean_hashtag}")
        
        # Sort by engagement and take top 2 for storage
        posts_sorted_by_engagement = sorted(
            scraped_posts, 
            key=lambda x: x.get("engagement", 0), 
            reverse=True
        )
        
        top_2_posts = posts_sorted_by_engagement[:2]
        
        print(f"📈 Engagement analysis complete. Top 2 posts selected from {len(scraped_posts)} scraped")
        debug_info["top_posts_engagement"] = [post.get("engagement", 0) for post in top_2_posts]
        
        # Save only top 2 to database
        saved_posts = await save_hashtag_posts_to_db(top_2_posts, clean_hashtag, db)
        
        # Save all scraped posts to JSON for trend analysis
        json_file = save_posts_to_json(scraped_posts, clean_hashtag, is_hashtag=True)
        debug_info["json_backup"] = json_file
        
        # Prepare response
        response_posts = [convert_post_to_response_format(post) for post in saved_posts]
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        print(f"✅ Successfully scraped and saved top {len(response_posts)} hashtag posts")
        
        return HashtagPostsResponse(
            success=True, 
            message=f"Scraped {len(scraped_posts)} posts, stored top {len(response_posts)} by engagement for #{clean_hashtag}",
            total_posts=len(response_posts), 
            posts=response_posts,
            from_cache=False, 
            execution_time_seconds=execution_time,
            debug_info=debug_info if debug else None
        )
        
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        error_message = str(e)
        
        print(f"❌ Error in hashtag posts endpoint: {error_message}")
        if debug:
            traceback.print_exc()
            debug_info["error_details"] = traceback.format_exc()
        
        # Ensure bot is closed even on error
        await _safe_close_bot(bot)
        
        return HashtagPostsResponse(
            success=False, 
            message=f"Error scraping hashtag posts: {error_message}", 
            total_posts=0, 
            posts=[], 
            from_cache=False, 
            execution_time_seconds=execution_time,
            debug_info=debug_info if debug else None
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "linkedin-scraper",
        "timestamp": datetime.utcnow().isoformat(),
        "data_directory": str(DATA_DIR),
        "environment": {
            "linkedin_email_configured": bool(os.getenv("LINKEDIN_EMAIL")),
            "linkedin_password_configured": bool(os.getenv("LINKEDIN_PASSWORD"))
        }
    }
