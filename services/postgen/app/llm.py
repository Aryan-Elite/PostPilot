from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from typing import List
import logging


from app.utils.prompt import (
    HYBRID_POST_TEMPLATE,
    SIMPLE_POST_TEMPLATE,
    QA_TEMPLATE,
    build_hybrid_prompt_values
)

logger = logging.getLogger(__name__)

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",  
    temperature=0.7,
    max_tokens=250,
    request_timeout=30
)

output_parser = StrOutputParser()

# Create chains
hybrid_chain = HYBRID_POST_TEMPLATE | llm | output_parser
simple_chain = SIMPLE_POST_TEMPLATE | llm | output_parser

def generate_hybrid_post(
    prompt: str, retriever, trending_posts: str,
    topic: str = None, tone: str = None, length: str = None, 
    audience: str = None, num_variations: int = 1
) -> List[str]:
    """
    Main function: Generate posts using RAG (user's posts) + trending posts
    
    Args:
        prompt: Main prompt text
        retriever: LangChain retriever for user's posts
        trending_posts: Scraped trending posts from hashtag
        topic: Topic for the post
        tone: Desired tone
        length: Desired length
        audience: Target audience
        num_variations: Number of variations to generate
        
    Returns:
        List of generated posts
    """
    if not prompt or not prompt.strip():
        logger.error("Empty prompt provided")
        return ["Error: No prompt provided"]
    
    if num_variations < 1 or num_variations > 10:
        num_variations = 1
    
    try:
        # Get user's similar posts via RAG
        relevant_docs = retriever.invoke(prompt)
        
        if relevant_docs:
            user_context = "\n\n".join([
                f"Previous post {i+1}: {doc.page_content}" 
                for i, doc in enumerate(relevant_docs[:5])
            ])
        else:
            user_context = "No similar previous posts found."
        
        # Build prompt values
        prompt_values = build_hybrid_prompt_values(
            prompt=prompt,
            topic=topic,
            tone=tone,
            length=length,
            audience=audience,
            user_context=user_context,
            trending_context=trending_posts
        )
        
        results = []
        for i in range(num_variations):
            try:
                logger.info(f"Generating hybrid variation {i+1} of {num_variations}")
                print(f"Generating variation {i+1} with RAG + trending context...")
                
                generated_text = hybrid_chain.invoke(prompt_values)
                
                if generated_text and generated_text.strip():
                    results.append(generated_text.strip())
                else:
                    results.append(f"Error: Empty response for variation {i+1}")
                    
            except Exception as e:
                logger.error(f"Error generating variation {i+1}: {str(e)}")
                results.append(f"Error generating variation {i+1}: {str(e)}")
        
        return results if results else ["Error: Failed to generate any variations"]
        
    except Exception as e:
        logger.error(f"Error in hybrid post generation: {e}")
        return [f"Error: {str(e)}"]

def generate_simple_post(prompt: str, num_variations: int = 1) -> List[str]:
    """
    Fallback function: Generate basic posts without context
    """
    if not prompt or not prompt.strip():
        return ["Error: No prompt provided"]
    
    if num_variations < 1 or num_variations > 10:
        num_variations = 1
    
    results = []
    for i in range(num_variations):
        try:
            generated_text = simple_chain.invoke({"prompt": prompt})
            
            if generated_text and generated_text.strip():
                results.append(generated_text.strip())
            else:
                results.append(f"Error: Empty response for variation {i+1}")
                
        except Exception as e:
            logger.error(f"Error generating simple variation {i+1}: {str(e)}")
            results.append(f"Error generating variation {i+1}: {str(e)}")
    
    return results if results else ["Error: Failed to generate any variations"]

def create_qa_chain(retriever):
    """
    Create a Q&A chain for document retrieval (if needed)
    """
    document_chain = create_stuff_documents_chain(llm, QA_TEMPLATE)
    return create_retrieval_chain(retriever, document_chain)