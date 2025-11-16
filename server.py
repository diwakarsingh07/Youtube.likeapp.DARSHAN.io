from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form, Cookie, Response, Request
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
import gridfs
from bson import ObjectId
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone, timedelta
import base64
from io import BytesIO
from PIL import Image
import requests

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# GridFS for video storage - use sync client for GridFS
from pymongo import MongoClient
sync_client = MongoClient(mongo_url)
sync_db = sync_client[os.environ['DB_NAME']]
fs = gridfs.GridFS(sync_db)

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Models
class User(BaseModel):
    id: str
    email: str
    name: str
    picture: str
    channel_name: Optional[str] = None
    subscribers: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserSession(BaseModel):
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Video(BaseModel):
    id: Optional[str] = None
    title: str
    description: str
    user_id: str
    channel_name: str
    channel_picture: str
    thumbnail: str  # base64
    video_file_id: str  # GridFS file ID
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    duration: Optional[int] = None  # seconds
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Comment(BaseModel):
    id: Optional[str] = None
    video_id: str
    user_id: str
    user_name: str
    user_picture: str
    text: str
    likes: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Subscription(BaseModel):
    subscriber_id: str
    channel_user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class VideoLike(BaseModel):
    video_id: str
    user_id: str
    is_like: bool  # True for like, False for dislike
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Auth endpoints
@api_router.post("/auth/session")
async def create_session(session_id: str, response: Response):
    """Exchange session_id for session_token"""
    try:
        # Call Emergent Auth API
        headers = {"X-Session-ID": session_id}
        auth_response = requests.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers=headers
        )
        
        if auth_response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid session")
        
        user_data = auth_response.json()
        session_token = user_data["session_token"]
        
        # Store user if not exists
        existing_user = await db.users.find_one({"email": user_data["email"]})
        if not existing_user:
            user = User(
                id=user_data["id"],
                email=user_data["email"],
                name=user_data["name"],
                picture=user_data["picture"],
                channel_name=user_data["name"]
            )
            await db.users.insert_one(user.dict())
        
        # Store session
        session = UserSession(
            user_id=user_data["id"],
            session_token=session_token,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7)
        )
        await db.user_sessions.insert_one(session.dict())
        
        # Set cookie
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=True,
            samesite="none",
            path="/",
            max_age=7*24*60*60
        )
        
        return {"user": user_data, "session_token": session_token}
    except Exception as e:
        logging.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/auth/me")
async def get_current_user(request: Request, session_token: Optional[str] = Cookie(None)):
    """Get current authenticated user"""
    # Try cookie first, then Authorization header
    token = session_token
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Verify session
    session = await db.user_sessions.find_one({
        "session_token": token,
        "expires_at": {"$gt": datetime.now(timezone.utc)}
    })
    
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    # Get user
    user = await db.users.find_one({"id": session["user_id"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response, session_token: Optional[str] = Cookie(None)):
    """Logout user"""
    token = session_token
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    if token:
        await db.user_sessions.delete_one({"session_token": token})
    
    response.delete_cookie("session_token", path="/")
    return {"message": "Logged out"}

# Video endpoints
@api_router.post("/videos/upload")
async def upload_video(
    request: Request,
    title: str = Form(...),
    description: str = Form(...),
    video: UploadFile = File(...),
    thumbnail: UploadFile = File(...),
    session_token: Optional[str] = Cookie(None)
):
    """Upload a video"""
    # Get current user
    user = await get_current_user(request, session_token)
    
    # Read video file and store in GridFS
    video_content = await video.read()
    video_file_id = fs.put(video_content, filename=video.filename, content_type=video.content_type)
    
    # Process thumbnail to base64
    thumbnail_content = await thumbnail.read()
    img = Image.open(BytesIO(thumbnail_content))
    img.thumbnail((320, 180))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    thumbnail_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Create video document
    video_doc = Video(
        id=str(ObjectId()),
        title=title,
        description=description,
        user_id=user["id"],
        channel_name=user["channel_name"] or user["name"],
        channel_picture=user["picture"],
        thumbnail=f"data:image/jpeg;base64,{thumbnail_base64}",
        video_file_id=str(video_file_id)
    )
    
    await db.videos.insert_one(video_doc.dict())
    return {"message": "Video uploaded", "video_id": video_doc.id}

@api_router.get("/videos")
async def get_videos(skip: int = 0, limit: int = 20):
    """Get video feed"""
    cursor = db.videos.find().sort("created_at", DESCENDING).skip(skip).limit(limit)
    videos = await cursor.to_list(length=limit)
    return videos

@api_router.get("/videos/{video_id}")
async def get_video(video_id: str):
    """Get single video details"""
    video = await db.videos.find_one({"id": video_id})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Increment view count
    await db.videos.update_one({"id": video_id}, {"$inc": {"views": 1}})
    video["views"] = video.get("views", 0) + 1
    
    return video

@api_router.get("/videos/{video_id}/stream")
async def stream_video(video_id: str):
    """Stream video file"""
    video = await db.videos.find_one({"id": video_id})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get video from GridFS
    try:
        file_id = ObjectId(video["video_file_id"])
        grid_out = fs.get(file_id)
        
        def iterfile():
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                data = grid_out.read(chunk_size)
                if not data:
                    break
                yield data
        
        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(grid_out.length)
            }
        )
    except Exception as e:
        logging.error(f"Video streaming error: {e}")
        raise HTTPException(status_code=500, detail="Error streaming video")

@api_router.get("/videos/search/{query}")
async def search_videos(query: str, skip: int = 0, limit: int = 20):
    """Search videos"""
    cursor = db.videos.find({
        "$or": [
            {"title": {"$regex": query, "$options": "i"}},
            {"description": {"$regex": query, "$options": "i"}},
            {"channel_name": {"$regex": query, "$options": "i"}}
        ]
    }).sort("created_at", DESCENDING).skip(skip).limit(limit)
    videos = await cursor.to_list(length=limit)
    return videos

@api_router.get("/videos/channel/{user_id}")
async def get_channel_videos(user_id: str, skip: int = 0, limit: int = 20):
    """Get videos by channel"""
    cursor = db.videos.find({"user_id": user_id}).sort("created_at", DESCENDING).skip(skip).limit(limit)
    videos = await cursor.to_list(length=limit)
    return videos

# Like/Dislike endpoints
@api_router.post("/videos/{video_id}/like")
async def like_video(video_id: str, is_like: bool, request: Request, session_token: Optional[str] = Cookie(None)):
    """Like or dislike a video"""
    user = await get_current_user(request, session_token)
    
    # Check existing like
    existing = await db.video_likes.find_one({"video_id": video_id, "user_id": user["id"]})
    
    if existing:
        # Update like/dislike
        old_is_like = existing["is_like"]
        await db.video_likes.update_one(
            {"video_id": video_id, "user_id": user["id"]},
            {"$set": {"is_like": is_like}}
        )
        
        # Update counts
        if old_is_like and not is_like:
            await db.videos.update_one({"id": video_id}, {"$inc": {"likes": -1, "dislikes": 1}})
        elif not old_is_like and is_like:
            await db.videos.update_one({"id": video_id}, {"$inc": {"likes": 1, "dislikes": -1}})
    else:
        # New like/dislike
        like_doc = VideoLike(video_id=video_id, user_id=user["id"], is_like=is_like)
        await db.video_likes.insert_one(like_doc.dict())
        
        if is_like:
            await db.videos.update_one({"id": video_id}, {"$inc": {"likes": 1}})
        else:
            await db.videos.update_one({"id": video_id}, {"$inc": {"dislikes": 1}})
    
    return {"message": "Updated"}

@api_router.get("/videos/{video_id}/like-status")
async def get_like_status(video_id: str, request: Request, session_token: Optional[str] = Cookie(None)):
    """Get user's like status for a video"""
    try:
        user = await get_current_user(request, session_token)
        like = await db.video_likes.find_one({"video_id": video_id, "user_id": user["id"]})
        return {"liked": like["is_like"] if like else None}
    except:
        return {"liked": None}

# Comment endpoints
@api_router.post("/videos/{video_id}/comments")
async def add_comment(video_id: str, text: str, request: Request, session_token: Optional[str] = Cookie(None)):
    """Add a comment"""
    user = await get_current_user(request, session_token)
    
    comment = Comment(
        id=str(ObjectId()),
        video_id=video_id,
        user_id=user["id"],
        user_name=user["name"],
        user_picture=user["picture"],
        text=text
    )
    
    await db.comments.insert_one(comment.dict())
    return comment

@api_router.get("/videos/{video_id}/comments")
async def get_comments(video_id: str, skip: int = 0, limit: int = 50):
    """Get comments for a video"""
    cursor = db.comments.find({"video_id": video_id}).sort("created_at", DESCENDING).skip(skip).limit(limit)
    comments = await cursor.to_list(length=limit)
    return comments

# Subscribe endpoints
@api_router.post("/channels/{channel_user_id}/subscribe")
async def subscribe(channel_user_id: str, request: Request, session_token: Optional[str] = Cookie(None)):
    """Subscribe to a channel"""
    user = await get_current_user(request, session_token)
    
    # Check if already subscribed
    existing = await db.subscriptions.find_one({
        "subscriber_id": user["id"],
        "channel_user_id": channel_user_id
    })
    
    if existing:
        # Unsubscribe
        await db.subscriptions.delete_one({"subscriber_id": user["id"], "channel_user_id": channel_user_id})
        await db.users.update_one({"id": channel_user_id}, {"$inc": {"subscribers": -1}})
        return {"subscribed": False}
    else:
        # Subscribe
        sub = Subscription(subscriber_id=user["id"], channel_user_id=channel_user_id)
        await db.subscriptions.insert_one(sub.dict())
        await db.users.update_one({"id": channel_user_id}, {"$inc": {"subscribers": 1}})
        return {"subscribed": True}

@api_router.get("/channels/{channel_user_id}/subscription-status")
async def get_subscription_status(channel_user_id: str, request: Request, session_token: Optional[str] = Cookie(None)):
    """Check if user is subscribed to channel"""
    try:
        user = await get_current_user(request, session_token)
        sub = await db.subscriptions.find_one({
            "subscriber_id": user["id"],
            "channel_user_id": channel_user_id
        })
        return {"subscribed": sub is not None}
    except:
        return {"subscribed": False}

@api_router.get("/channels/{user_id}")
async def get_channel(user_id: str):
    """Get channel details"""
    user = await db.users.find_one({"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    # Get video count
    video_count = await db.videos.count_documents({"user_id": user_id})
    
    return {
        **user,
        "video_count": video_count
    }

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
