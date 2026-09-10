import logging
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from config import MONGO_URI, DATABASE_NAME

import certifi

logger = logging.getLogger("medassist.database")

class DatabaseManager:
    client: MongoClient = None
    db: Database = None

    @classmethod
    def connect(cls) -> Database:
        if cls.client is None:
            try:
                cls.client = MongoClient(
                    MONGO_URI,
                    tlsCAFile=certifi.where(),
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000
                )
                cls.db = cls.client[DATABASE_NAME]
                # Ping database to verify connection
                cls.client.admin.command("ping")
                logger.info(f"Connected to MongoDB Atlas: {DATABASE_NAME}")
                cls._init_indexes()
            except Exception as e:
                logger.warning(f"Standard SSL connection failed ({e}), attempting resilient TLS fallback...")
                try:
                    cls.client = MongoClient(
                        MONGO_URI,
                        tls=True,
                        tlsAllowInvalidCertificates=True,
                        serverSelectionTimeoutMS=5000,
                        connectTimeoutMS=5000,
                        socketTimeoutMS=5000
                    )
                    cls.db = cls.client[DATABASE_NAME]
                    cls.client.admin.command("ping")
                    logger.info(f"Connected to MongoDB Atlas via TLS fallback: {DATABASE_NAME}")
                    cls._init_indexes()
                except Exception as e2:
                    logger.error(f"Failed to connect to MongoDB: {e2}")
                    raise e2
        return cls.db

    @classmethod
    def _init_indexes(cls):
        try:
            # Create unique index on email for patients and doctors
            cls.db["patients"].create_index("email", unique=True)
            cls.db["doctors"].create_index("email", unique=True)
            cls.db["symptoms_log"].create_index("email")
            cls.db["symptoms_log"].create_index("created_at")
            cls.db["appointments"].create_index("patient_email")
            cls.db["appointments"].create_index("doctor_email")
            logger.info("MongoDB indexes verified successfully.")
        except Exception as e:
            logger.warning(f"Index creation notice: {e}")

    @classmethod
    def get_db(cls) -> Database:
        if cls.db is None:
            return cls.connect()
        return cls.db

    @classmethod
    def get_patients_collection(cls) -> Collection:
        return cls.get_db()["patients"]

    @classmethod
    def get_doctors_collection(cls) -> Collection:
        return cls.get_db()["doctors"]

    @classmethod
    def get_symptoms_log_collection(cls) -> Collection:
        return cls.get_db()["symptoms_log"]

    @classmethod
    def get_appointments_collection(cls) -> Collection:
        return cls.get_db()["appointments"]

    @classmethod
    def get_sample_diseases_collection(cls) -> Collection:
        return cls.get_db()["Sample_diseases"]

    @classmethod
    def get_severity_collection(cls) -> Collection:
        return cls.get_db()["Severity"]

    @classmethod
    def get_categories_collection(cls) -> Collection:
        return cls.get_db()["categories"]

    @classmethod
    def close(cls):
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            logger.info("MongoDB connection closed.")
