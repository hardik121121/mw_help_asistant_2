# 🔑 How to Get Your API Keys

You need to obtain API keys from 4 services. Here's how:

## 1. OpenAI API Key (Required)
**Purpose**: Embeddings (text-embedding-3-large) and query decomposition

**Steps**:
1. Go to: https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-proj-...`)
5. **Important**: Add $5-10 credit to your account for embeddings

**Cost**: ~$3-5 one-time for processing PDF, ~$0.0001 per query

---

## 2. Pinecone API Key (Required)
**Purpose**: Vector database for storing embeddings

**Steps**:
1. Go to: https://app.pinecone.io/
2. Sign up for free account
3. Go to "API Keys" in dashboard
4. Copy your API key
5. Note your environment (e.g., `gcp-starter` or `us-east-1-aws`)

**Cost**: FREE (100K vectors free tier - enough for this project)

---

## 3. Cohere API Key (Required)
**Purpose**: Re-ranking retrieved results for better precision

**Steps**:
1. Go to: https://dashboard.cohere.com/api-keys
2. Sign up for free account
3. Go to API Keys section
4. Copy your API key

**Cost**: FREE (1,000 re-rank calls/month free tier)

---

## 4. Groq API Key (Required)
**Purpose**: LLM generation (Llama 3.3 70B)

**Steps**:
1. Go to: https://console.groq.com/keys
2. Sign up for free account
3. Create a new API key
4. Copy the key (starts with `gsk_...`)

**Cost**: FREE (14,400 requests/day free tier)

---

## 📝 Adding Keys to .env

Once you have all 4 keys, edit the `.env` file:

```bash
# Method 1: Use a text editor
nano .env

# Method 2: Use VS Code
code .env

# Method 3: Use vim
vim .env
```

Replace the placeholder values:

```bash
# Before (example):
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# After (your real key):
OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_KEY_HERE
```

**Do this for all 4 keys**: OPENAI_API_KEY, PINECONE_API_KEY, COHERE_API_KEY, GROQ_API_KEY

---

## ✅ Verify Your Setup

After adding all keys, run:

```bash
./venv/bin/python config/settings.py
```

You should see:
```
✓ OpenAI: Set
✓ Pinecone: Set
✓ Cohere: Set
✓ Groq: Set
```

---

## 💰 Total Cost Estimate

| Service | One-time | Per Query | Free Tier |
|---------|----------|-----------|-----------|
| OpenAI | $3-5 | $0.0001 | No free tier for embeddings |
| Pinecone | $0 | $0 | 100K vectors free |
| Cohere | $0 | $0.002 (after 1K) | 1,000 calls/month |
| Groq | $0 | $0 | 14,400 req/day |
| **Total** | **$3-5** | **~$0.002** | **Generous free tiers** |

**Monthly cost** (300 queries): ~$10-15

---

## ⚠️ Important Notes

1. **Keep keys secret**: Never commit .env to git (already in .gitignore)
2. **OpenAI credit**: You need to add credit to your OpenAI account
3. **Free tiers**: All except OpenAI have generous free tiers
4. **Rate limits**: Groq has best free tier (14.4K requests/day)

---

## 🆘 Having Trouble?

- OpenAI requires credit card for API access
- Pinecone free tier is serverless (limited to specific regions)
- Cohere free tier is 1K calls/month (then $0.002/call)
- Groq is completely free for high usage

---

## Next Steps

After adding all API keys:
1. Run configuration validation: `./venv/bin/python config/settings.py`
2. If all keys are valid, proceed to Phase 2 testing
3. Process your PDF and start building!
