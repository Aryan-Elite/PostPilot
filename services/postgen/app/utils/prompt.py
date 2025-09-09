from langchain_core.prompts import PromptTemplate

HYBRID_POST_TEMPLATE = PromptTemplate(
    input_variables=["prompt", "topic", "tone", "length", "audience", "user_context", "trending_context"],
    template=(
        "Generate a LinkedIn post: {prompt}\n"
        "Topic: {topic}\n"
        "Tone: {tone}\n" 
        "Length: {length}\n"
        "Target audience: {audience}\n\n"
        
        "CONTEXT - Your Previous Similar Posts:\n"
        "{user_context}\n\n"
        
        "CONTEXT - Current Trending Posts:\n" 
        "{trending_context}\n\n"
        
        "Instructions:\n"
        "- Create original content inspired by both contexts\n"
        "- Match your previous writing style from similar posts\n"
        "- Incorporate current trends from trending posts\n"
        "- Include relevant hashtags\n"
        "- Keep it authentic and engaging\n\n"
        
        "Return only the final post text:"
    )
)

# Simple fallback template
SIMPLE_POST_TEMPLATE = PromptTemplate(
    input_variables=["prompt"],
    template="{prompt}"
)

# Q&A template for retrieval chains
QA_TEMPLATE = PromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    
    Question: {input}
    
    Answer:"""
)


def build_hybrid_prompt_values(
    prompt: str, topic: str = None, tone: str = None, length: str = None, 
    audience: str = None, user_context: str = None, trending_context: str = None
) -> dict:
    """
    Build prompt values for hybrid generation (RAG + Trending)
    
    Args:
        prompt: Main prompt text
        topic: Topic for the post  
        tone: Desired tone
        length: Desired length
        audience: Target audience
        user_context: User's previous similar posts from RAG
        trending_context: Scraped trending posts
        
    Returns:
        Dictionary of prompt values with empty string defaults
    """
    return {
        "prompt": prompt or "",
        "topic": topic or "",
        "tone": tone or "professional", 
        "length": length or "medium",
        "audience": audience or "",
        "user_context": user_context or "No previous posts found.",
        "trending_context": trending_context or "No trending posts available."
    }