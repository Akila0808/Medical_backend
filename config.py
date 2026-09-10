import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://anithaparames1808_db_user:ju5viHnzqAPPHU0A@cluster0.txlrfgr.mongodb.net/?appName=Cluster0"
)
DATABASE_NAME = os.getenv("DATABASE_NAME", "Medical_assistant")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "medassist_super_secret_jwt_key_2026_secure_clinical")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "24"))

MODEL_PATH = os.getenv("MODEL_PATH", "medassist_disease_model.pkl")
MODEL_FEATURES_PATH = os.getenv("MODEL_FEATURES_PATH", "model_features.pkl")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
