
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(title="Prof Ido API")

# Untuk pengujian lokal saja.
# Saat online, ubah menjadi domain frontend milikmu.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None


class Message(BaseModel):
    role: str
    text: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)


@app.get("/")
def home():
    return {"status": "Prof Ido backend aktif"}


@app.post("/chat")
def chat(data: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="API key belum diatur pada backend."
        )

    message = data.message.strip()
    if not message:
        raise HTTPException(
            status_code=400,
            detail="Pesan tidak boleh kosong."
        )

    contents = []

    # Terima hanya peran yang dikenali.
    for item in data.history[-12:]:
        if item.role not in ("user", "model", "bot"):
            continue

        role = "user" if item.role == "user" else "model"
        previous_text = item.text.strip()

        if previous_text:
            contents.append(
                types.Content(
                    role=role,
                    parts=[
                        types.Part.from_text(
                            text=previous_text[:2000]
                        )
                    ],
                )
            )

    # Pesan terbaru ditambahkan satu kali.
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)],
        )
    )

    try:
        result = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Nama Anda Prof Ido, asisten digital yang "
                    "informatif, hangat, dan profesional. "
                    "Gunakan bahasa Indonesia formal yang alami "
                    "jika pengguna berbicara dalam bahasa Indonesia. "
                    "Perhatikan konteks percakapan sebelumnya. "
                    "Jelaskan dengan jelas, akui jika tidak yakin, "
                    "dan jangan mengarang fakta."
                ),
                max_output_tokens=900,
            ),
        )

        answer = result.text
        if not answer:
            raise ValueError("Model memberikan jawaban kosong.")

        return {"reply": answer}

    except Exception:
        # Detail internal tidak ditampilkan kepada pengunjung.
        raise HTTPException(
            status_code=502,
            detail=(
                "AI gagal menjawab. Periksa nama model, API key, "
                "koneksi internet, dan kuota."
            ),
        )
