import hashlib
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

from app.db import get_db


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mongo_client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = mongo_client[os.getenv("MONGO_DB")]
reviews_collection = mongo_db["reviews"]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding


def fetch_all(cursor, query: str, params: tuple):
    cursor.execute(query, params)
    return cursor.fetchall()


def fetch_one(cursor, query: str, params: tuple):
    cursor.execute(query, params)
    return cursor.fetchone()


def fetch_dirty_contact_embedding_ids(limit: int, embedding_ids: list[int] | None = None):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    try:
        params: list[int] = []
        id_filter = ""

        if embedding_ids:
            placeholders = ", ".join(["%s"] * len(embedding_ids))
            id_filter = f" AND id IN ({placeholders})"
            params.extend(embedding_ids)

        params.append(limit)
        cursor.execute(
            f"""
            SELECT id
            FROM user_contact_embeddings
            WHERE needs_rebuild = 1
            {id_filter}
            ORDER BY id
            LIMIT %s
            """,
            tuple(params)
        )

        return [row["id"] for row in cursor.fetchall()]
    finally:
        cursor.close()
        db.close()


def rebuild_dirty_contact_embeddings(limit: int = 100, embedding_ids: list[int] | None = None):
    limit = max(1, min(limit, 500))
    dirty_ids = fetch_dirty_contact_embedding_ids(limit, embedding_ids)

    summary = {
        "requested": len(dirty_ids),
        "rebuilt": 0,
        "skipped": 0,
        "errors": []
    }

    for embedding_id in dirty_ids:
        try:
            result = rebuild_contact_embedding(embedding_id)

            if result.get("message") == "Contact embedding rebuilt successfully":
                summary["rebuilt"] += 1
            else:
                summary["skipped"] += 1
        except Exception as exc:
            summary["errors"].append({
                "embedding_id": embedding_id,
                "error": str(exc)
            })

    return summary


def rebuild_contact_embedding(embedding_id: int):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT
                uce.id AS embedding_id,
                uce.user_id,
                uce.contact_id,
                c.phone,
                uc.id AS user_contact_id,
                uc.display_name,
                cu.id AS contact_user_id,
                cu.fname,
                cu.lname
            FROM user_contact_embeddings uce
            JOIN contacts c ON c.id = uce.contact_id
            LEFT JOIN user_contacts uc
                   ON uc.user_id = uce.user_id
                  AND uc.contact_id = uce.contact_id
            LEFT JOIN users cu
                   ON cu.phone = c.phone
            WHERE uce.id = %s
              AND uce.needs_rebuild = 1
            """,
            (embedding_id,)
        )

        row = cursor.fetchone()

        if not row:
            return {
                "message": "No contact embedding found that needs rebuild",
                "embedding_id": embedding_id
            }

        if row["display_name"]:
            name = row["display_name"]
        elif row["fname"] or row["lname"]:
            name = f"{row['fname'] or ''} {row['lname'] or ''}".strip()
        else:
            name = "Unknown"

        default_identity = "None"
        if row["contact_user_id"]:
            rows_dd = fetch_all(
                cursor,
                """
                SELECT label, description
                FROM default_description
                WHERE users_id = %s
                """,
                (row["contact_user_id"],)
            )

            if rows_dd:
                default_identity = "\n".join(
                    f"- {d['label']}: {d['description']}"
                    for d in rows_dd
                )

        personal_labels = "None yet"
        if row["user_contact_id"]:
            rows_pl = fetch_all(
                cursor,
                """
                SELECT label, description
                FROM user_contact_descriptions
                WHERE user_contact_id = %s
                """,
                (row["user_contact_id"],)
            )

            if rows_pl:
                personal_labels = "\n".join(
                    f"- {p['label']}: {p['description']}"
                    for p in rows_pl
                )

        cv_text = "None"
        if row["contact_user_id"]:
            cv = fetch_one(
                cursor,
                """
                SELECT cv
                FROM users_cv
                WHERE user_id = %s
                """,
                (row["contact_user_id"],)
            )

            if cv and cv["cv"]:
                cv_text = cv["cv"]

        reviews_text = "None"

        if row["contact_user_id"]:
            desc_rows = fetch_all(
                cursor,
                """
                SELECT id, label, description
                FROM default_description
                WHERE users_id = %s
                """,
                (row["contact_user_id"],)
            )

            formatted_reviews = []

            for desc in desc_rows:
                mongo_reviews = reviews_collection.find({
                    "default_description_id": desc["id"]
                })

                for rv in mongo_reviews:
                    if rv.get("review"):
                        formatted_reviews.append(
                            f"""[REVIEW CONTEXT]
Role: {desc['label'] or 'Unknown'}
Description: {desc['description'] or 'None'}
Review: {rv['review']}
"""
                        )

            if formatted_reviews:
                reviews_text = "\n".join(formatted_reviews)

        profile_text = f"""CONTACT
Name: {name}
Phone: {row['phone']}

DEFAULT IDENTITY
{default_identity}

MY PERSONAL LABELS
{personal_labels}

CV
{cv_text}

REVIEWS
{reviews_text}
"""

        context_hash = sha256(profile_text)
        embedding = embed(profile_text)

        cursor.execute(
            """
            UPDATE user_contact_embeddings
            SET profile_text = %s,
                embedding = %s,
                context_hash = %s,
                needs_rebuild = 0
            WHERE id = %s
            """,
            (
                profile_text,
                json.dumps(embedding),
                context_hash,
                row["embedding_id"]
            )
        )

        db.commit()

        return {
            "message": "Contact embedding rebuilt successfully",
            "embedding_id": row["embedding_id"],
            "contact_id": row["contact_id"],
            "user_id": row["user_id"],
            "name": name
        }
    finally:
        cursor.close()
        db.close()
