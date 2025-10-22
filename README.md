# 🎥 YouTube Fear & Greed Index (AI-Powered)

A Streamlit app that scans any YouTube channel’s video titles for a given month and uses OpenAI’s GPT models to estimate overall **market sentiment** — whether the tone of the videos leans **Bullish**, **Bearish**, or **Neutral**.

Think of it as a little sentiment meter for the crypto crowd — or anyone who wants to know how optimistic (or fearful) a YouTube channel’s content has been lately.

---

## ✨ What It Does

1. **Fetches YouTube videos** from any public channel using the official YouTube Data API.  
2. **Classifies each video title** with GPT-4 or GPT-4o-mini into  
   🟢 Bullish 🔴 Bearish ⚪ Neutral.  
3. **Calculates a sentiment index** from −1 to +1:  
   - −1 = fully Bearish  
   - 0 = Neutral  
   - +1 = fully Bullish  
4. **Visualizes the result** as a simple “Fear & Greed” meter.  
   - < 0 → Fear (red)  
   - 0 – 0.4 → Neutral (gray)  
   - > 0.4 → Greed (green)  
5. Lets you **download** the data in CSV or Excel format.

---

