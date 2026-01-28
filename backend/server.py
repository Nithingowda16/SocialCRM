from fastapi import FastAPI, APIRouter, HTTPException, Response, Request, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import httpx
from emergentintegrations.llm.chat import LlmChat, UserMessage


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= Models =============
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: EmailStr
    name: str
    picture: Optional[str] = None
    created_at: datetime


class UserSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    session_id: str
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime


class SocialAccount(BaseModel):
    model_config = ConfigDict(extra="ignore")
    account_id: str
    user_id: str
    platform: str  # facebook, instagram, linkedin
    platform_user_id: str
    platform_username: str
    access_token: str
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    account_name: str
    profile_picture: Optional[str] = None
    connected_at: datetime
    is_active: bool = True


class SocialAccountCreate(BaseModel):
    platform: str
    access_token: str
    refresh_token: Optional[str] = None


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message_id: str
    user_id: str
    account_id: str
    platform: str
    platform_message_id: str
    message_type: str  # message, comment
    sender_id: str
    sender_name: str
    sender_avatar: Optional[str] = None
    content: str
    parent_message_id: Optional[str] = None
    post_id: Optional[str] = None
    is_read: bool = False
    replied: bool = False
    assigned_to: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class MessageReply(BaseModel):
    message_id: str
    reply_text: str


class InternalNote(BaseModel):
    model_config = ConfigDict(extra="ignore")
    note_id: str
    user_id: str
    message_id: str
    author_id: str
    author_name: str
    content: str
    created_at: datetime


class InternalNoteCreate(BaseModel):
    message_id: str
    content: str


class ScheduledPost(BaseModel):
    model_config = ConfigDict(extra="ignore")
    post_id: str
    user_id: str
    account_id: str
    platform: str
    content: str
    media_urls: Optional[List[str]] = []
    scheduled_for: datetime
    status: str  # pending, published, failed
    created_at: datetime


class ScheduledPostCreate(BaseModel):
    account_id: str
    platform: str
    content: str
    media_urls: Optional[List[str]] = []
    scheduled_for: datetime


class AIReplyRequest(BaseModel):
    message_content: str
    conversation_history: Optional[List[Dict[str, str]]] = []


class AssignConversation(BaseModel):
    message_id: str
    assigned_to: str


# ============= Authentication Helper =============
async def get_current_user(request: Request) -> User:
    """Extract user from session token (cookie or Authorization header)"""
    session_token = None
    
    # Try cookie first
    session_token = request.cookies.get("session_token")
    
    # Fallback to Authorization header
    if not session_token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.replace("Bearer ", "")
    
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Find session
    session_doc = await db.user_sessions.find_one(
        {"session_token": session_token},
        {"_id": 0}
    )
    
    if not session_doc:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check expiry
    expires_at = session_doc["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Get user
    user_doc = await db.users.find_one(
        {"user_id": session_doc["user_id"]},
        {"_id": 0}
    )
    
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Parse datetime if string
    if isinstance(user_doc.get('created_at'), str):
        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
    
    return User(**user_doc)


# ============= Auth Routes =============
@api_router.post("/auth/session")
async def create_session(request: Request, response: Response):
    """Exchange session_id from Emergent Auth for user data and create session"""
    body = await request.json()
    session_id = body.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    
    # Call Emergent Auth API
    async with httpx.AsyncClient() as client:
        try:
            auth_response = await client.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={"X-Session-ID": session_id}
            )
            auth_response.raise_for_status()
            auth_data = auth_response.json()
        except Exception as e:
            logger.error(f"Failed to get session data: {e}")
            raise HTTPException(status_code=401, detail="Invalid session_id")
    
    # Check if user exists
    user_doc = await db.users.find_one(
        {"email": auth_data["email"]},
        {"_id": 0}
    )
    
    if user_doc:
        # Update existing user
        user_id = user_doc["user_id"]
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {
                "name": auth_data["name"],
                "picture": auth_data.get("picture")
            }}
        )
    else:
        # Create new user
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user_doc = {
            "user_id": user_id,
            "email": auth_data["email"],
            "name": auth_data["name"],
            "picture": auth_data.get("picture"),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.users.insert_one(user_doc)
    
    # Create session
    session_token = auth_data["session_token"]
    session_doc = {
        "session_id": f"session_{uuid.uuid4().hex[:12]}",
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.user_sessions.insert_one(session_doc)
    
    # Set httpOnly cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=7*24*60*60
    )
    
    # Return user data
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    if isinstance(user_doc.get('created_at'), str):
        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])
    
    return User(**user_doc)


@api_router.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user from session"""
    return current_user


@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Logout and clear session"""
    session_token = request.cookies.get("session_token")
    
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    
    response.delete_cookie(key="session_token", path="/")
    return {"message": "Logged out successfully"}


# ============= Social Account Routes =============
@api_router.get("/social-accounts", response_model=List[SocialAccount])
async def get_social_accounts(current_user: User = Depends(get_current_user)):
    """Get all connected social accounts for current user"""
    accounts = await db.social_accounts.find(
        {"user_id": current_user.user_id, "is_active": True},
        {"_id": 0}
    ).to_list(100)
    
    for account in accounts:
        if isinstance(account.get('connected_at'), str):
            account['connected_at'] = datetime.fromisoformat(account['connected_at'])
        if isinstance(account.get('token_expires_at'), str):
            account['token_expires_at'] = datetime.fromisoformat(account['token_expires_at'])
    
    return accounts


@api_router.post("/social-accounts", response_model=SocialAccount)
async def connect_social_account(
    account_data: SocialAccountCreate,
    current_user: User = Depends(get_current_user)
):
    """Connect a new social media account"""
    # In a real implementation, validate the token with the platform API
    # For now, we'll create a mock account
    account_id = f"account_{uuid.uuid4().hex[:12]}"
    
    account_doc = {
        "account_id": account_id,
        "user_id": current_user.user_id,
        "platform": account_data.platform,
        "platform_user_id": f"{account_data.platform}_{uuid.uuid4().hex[:8]}",
        "platform_username": f"user_{account_data.platform}",
        "access_token": account_data.access_token,
        "refresh_token": account_data.refresh_token,
        "token_expires_at": (datetime.now(timezone.utc) + timedelta(days=60)).isoformat(),
        "account_name": f"My {account_data.platform.title()} Account",
        "profile_picture": None,
        "connected_at": datetime.now(timezone.utc).isoformat(),
        "is_active": True
    }
    
    await db.social_accounts.insert_one(account_doc)
    
    account_doc['connected_at'] = datetime.fromisoformat(account_doc['connected_at'])
    account_doc['token_expires_at'] = datetime.fromisoformat(account_doc['token_expires_at'])
    
    return SocialAccount(**account_doc)


@api_router.delete("/social-accounts/{account_id}")
async def disconnect_social_account(
    account_id: str,
    current_user: User = Depends(get_current_user)
):
    """Disconnect a social media account"""
    result = await db.social_accounts.update_one(
        {"account_id": account_id, "user_id": current_user.user_id},
        {"$set": {"is_active": False}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Account not found")
    
    return {"message": "Account disconnected"}


# ============= Messages Routes =============
@api_router.get("/messages", response_model=List[Message])
async def get_messages(
    platform: Optional[str] = None,
    message_type: Optional[str] = None,
    is_read: Optional[bool] = None,
    assigned_to: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get all messages/comments from connected accounts"""
    query = {"user_id": current_user.user_id}
    
    if platform:
        query["platform"] = platform
    if message_type:
        query["message_type"] = message_type
    if is_read is not None:
        query["is_read"] = is_read
    if assigned_to:
        query["assigned_to"] = assigned_to
    
    messages = await db.messages.find(query, {"_id": 0}).sort("created_at", -1).to_list(500)
    
    for msg in messages:
        if isinstance(msg.get('created_at'), str):
            msg['created_at'] = datetime.fromisoformat(msg['created_at'])
        if isinstance(msg.get('updated_at'), str):
            msg['updated_at'] = datetime.fromisoformat(msg['updated_at'])
    
    return messages


@api_router.post("/messages/{message_id}/reply")
async def reply_to_message(
    message_id: str,
    reply_data: MessageReply,
    current_user: User = Depends(get_current_user)
):
    """Reply to a message or comment"""
    # Find the message
    message = await db.messages.find_one(
        {"message_id": message_id, "user_id": current_user.user_id},
        {"_id": 0}
    )
    
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # In a real implementation, send the reply via the platform API
    # For now, just mark as replied
    await db.messages.update_one(
        {"message_id": message_id},
        {"$set": {
            "replied": True,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    return {"message": "Reply sent successfully"}


@api_router.post("/messages/{message_id}/assign")
async def assign_message(
    message_id: str,
    assign_data: AssignConversation,
    current_user: User = Depends(get_current_user)
):
    """Assign a message to a team member"""
    result = await db.messages.update_one(
        {"message_id": message_id, "user_id": current_user.user_id},
        {"$set": {"assigned_to": assign_data.assigned_to}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return {"message": "Conversation assigned"}


@api_router.patch("/messages/{message_id}/read")
async def mark_message_read(
    message_id: str,
    current_user: User = Depends(get_current_user)
):
    """Mark a message as read"""
    result = await db.messages.update_one(
        {"message_id": message_id, "user_id": current_user.user_id},
        {"$set": {"is_read": True}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return {"message": "Message marked as read"}


# ============= AI Reply Routes =============
@api_router.post("/ai/suggest-reply")
async def suggest_reply(
    request_data: AIReplyRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate AI-powered reply suggestions using Claude Sonnet 4.5"""
    try:
        # Initialize LlmChat with Claude Sonnet 4.5
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=f"ai_reply_{current_user.user_id}_{uuid.uuid4().hex[:8]}",
            system_message="You are a helpful customer service assistant. Generate professional, empathetic, and concise replies to customer messages. Keep responses friendly and solution-oriented."
        ).with_model("anthropic", "claude-sonnet-4-5-20250929")
        
        # Build context from conversation history
        context = ""
        if request_data.conversation_history:
            context = "Previous conversation:\n"
            for msg in request_data.conversation_history[-5:]:  # Last 5 messages
                context += f"{msg.get('sender', 'User')}: {msg.get('content', '')}\n"
            context += "\n"
        
        prompt = f"{context}Customer message: {request_data.message_content}\n\nGenerate 2-3 professional reply options (each on a new line, numbered)."
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        return {"suggestions": response}
    except Exception as e:
        logger.error(f"AI reply error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate AI reply")


# ============= Internal Notes Routes =============
@api_router.get("/notes/{message_id}", response_model=List[InternalNote])
async def get_notes(
    message_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get internal notes for a message"""
    notes = await db.internal_notes.find(
        {"message_id": message_id, "user_id": current_user.user_id},
        {"_id": 0}
    ).sort("created_at", -1).to_list(100)
    
    for note in notes:
        if isinstance(note.get('created_at'), str):
            note['created_at'] = datetime.fromisoformat(note['created_at'])
    
    return notes


@api_router.post("/notes", response_model=InternalNote)
async def create_note(
    note_data: InternalNoteCreate,
    current_user: User = Depends(get_current_user)
):
    """Add an internal note to a message"""
    note_id = f"note_{uuid.uuid4().hex[:12]}"
    
    note_doc = {
        "note_id": note_id,
        "user_id": current_user.user_id,
        "message_id": note_data.message_id,
        "author_id": current_user.user_id,
        "author_name": current_user.name,
        "content": note_data.content,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.internal_notes.insert_one(note_doc)
    
    note_doc['created_at'] = datetime.fromisoformat(note_doc['created_at'])
    
    return InternalNote(**note_doc)


# ============= Scheduled Posts Routes =============
@api_router.get("/scheduled-posts", response_model=List[ScheduledPost])
async def get_scheduled_posts(
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get all scheduled posts"""
    query = {"user_id": current_user.user_id}
    if status:
        query["status"] = status
    
    posts = await db.scheduled_posts.find(query, {"_id": 0}).sort("scheduled_for", 1).to_list(100)
    
    for post in posts:
        if isinstance(post.get('created_at'), str):
            post['created_at'] = datetime.fromisoformat(post['created_at'])
        if isinstance(post.get('scheduled_for'), str):
            post['scheduled_for'] = datetime.fromisoformat(post['scheduled_for'])
    
    return posts


@api_router.post("/scheduled-posts", response_model=ScheduledPost)
async def create_scheduled_post(
    post_data: ScheduledPostCreate,
    current_user: User = Depends(get_current_user)
):
    """Schedule a new post"""
    post_id = f"post_{uuid.uuid4().hex[:12]}"
    
    post_doc = {
        "post_id": post_id,
        "user_id": current_user.user_id,
        "account_id": post_data.account_id,
        "platform": post_data.platform,
        "content": post_data.content,
        "media_urls": post_data.media_urls or [],
        "scheduled_for": post_data.scheduled_for.isoformat() if isinstance(post_data.scheduled_for, datetime) else post_data.scheduled_for,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.scheduled_posts.insert_one(post_doc)
    
    post_doc['created_at'] = datetime.fromisoformat(post_doc['created_at'])
    if isinstance(post_doc['scheduled_for'], str):
        post_doc['scheduled_for'] = datetime.fromisoformat(post_doc['scheduled_for'])
    
    return ScheduledPost(**post_doc)


@api_router.delete("/scheduled-posts/{post_id}")
async def delete_scheduled_post(
    post_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a scheduled post"""
    result = await db.scheduled_posts.delete_one(
        {"post_id": post_id, "user_id": current_user.user_id}
    )
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Post not found")
    
    return {"message": "Scheduled post deleted"}


# ============= Analytics Routes =============
@api_router.get("/analytics/overview")
async def get_analytics(
    days: int = 7,
    current_user: User = Depends(get_current_user)
):
    """Get analytics overview"""
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    # Get total messages
    total_messages = await db.messages.count_documents({
        "user_id": current_user.user_id,
        "created_at": {"$gte": start_date.isoformat()}
    })
    
    # Get replied messages
    replied_count = await db.messages.count_documents({
        "user_id": current_user.user_id,
        "replied": True,
        "created_at": {"$gte": start_date.isoformat()}
    })
    
    # Get unread messages
    unread_count = await db.messages.count_documents({
        "user_id": current_user.user_id,
        "is_read": False
    })
    
    # Platform breakdown
    pipeline = [
        {"$match": {
            "user_id": current_user.user_id,
            "created_at": {"$gte": start_date.isoformat()}
        }},
        {"$group": {
            "_id": "$platform",
            "count": {"$sum": 1}
        }}
    ]
    
    platform_stats = await db.messages.aggregate(pipeline).to_list(10)
    
    return {
        "total_messages": total_messages,
        "replied_messages": replied_count,
        "unread_messages": unread_count,
        "response_rate": round((replied_count / total_messages * 100) if total_messages > 0 else 0, 2),
        "platform_breakdown": [
            {"platform": stat["_id"], "count": stat["count"]}
            for stat in platform_stats
        ],
        "date_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": days
        }
    }


# ============= Test Route =============
@api_router.get("/")
async def root():
    return {"message": "Social CRM API"}


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
