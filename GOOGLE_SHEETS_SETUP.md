# Google Sheets Integration Setup Guide

## ✅ Installation Complete!

The required libraries (`gspread` and `google-auth`) have been installed, and your `google_credentials.json` file is detected.

---

## 🔧 Quick Setup Steps

### 1. **Create or Open Your Google Sheet**
   - Go to [Google Sheets](https://sheets.google.com)
   - Create a new spreadsheet (or use an existing one)
   - Copy the **Sheet ID** from the URL:
     ```
     https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID_HERE/edit
                                              ^^^^^^^^^^^^^^^^^^^
     ```

### 2. **Share the Sheet with Your Service Account**
   - Open your `google_credentials.json` file
   - Find the `client_email` field (looks like: `xxxxx@xxxxx.iam.gserviceaccount.com`)
   - In your Google Sheet, click **Share** button
   - Add the service account email as an **Editor**
   - Click **Send**

### 3. **Configure the App**

#### Option A: Using .env File (Recommended)
1. Open your `.env` file in the project root
2. Add this line (replace with your actual Sheet ID):
   ```bash
   GOOGLE_SHEET_ID=your_sheet_id_here
   ```
3. Save the file
4. Restart the Streamlit app (it will auto-connect)

#### Option B: Using the App UI
1. Open the Streamlit app
2. Look at the sidebar under "📊 Google Sheets Integration"
3. It should now show "✅ Google Sheets API Ready"
4. Paste your Sheet ID in the text box
5. Click "🔗 Connect to Sheet"

---

## 📊 What Gets Synced

Once connected, the app will automatically sync:

✅ **Portfolio Analysis Results**
   - Ticker, Score, Recommendation
   - Agent scores breakdown
   - Current price, P/E ratio, beta
   - Analysis timestamp

✅ **QA & Learning Center Data**
   - Historical analyses for tracking
   - Score changes over time
   - Learning insights

---

## ⚙️ Configuration Options

In the sidebar, you'll see:

- **🔄 Auto-update on analysis** - Automatically push new results to Sheets
- **📄 Open Sheet** - Direct link to your Google Sheet
- **🔄 Sync QA Analyses Now** - Manual sync button

---

## 🎯 Expected Sidebar Status

After completing setup, you should see:

```
📊 Google Sheets Integration
✅ Google Sheets API Ready

Google Sheet ID: [your_sheet_id]
[Already connected - see link below]

☑️ Auto-update on analysis
📄 Open Sheet
🔄 Sync QA Analyses Now
```

---

## 🐛 Troubleshooting

### If you see "⚙️ Not configured (optional)"

**Cause:** Missing libraries or credentials

**Solution:**
1. Check `google_credentials.json` exists in project root ✅ (DONE)
2. Install required libraries: ✅ (DONE)
   ```bash
   pip install gspread google-auth
   ```
3. Restart Streamlit app ✅ (DONE)

### If you see "❌ Connection failed"

**Possible causes:**
- Invalid Sheet ID
- Sheet not shared with service account email
- Service account doesn't have Editor permissions

**Solution:**
1. Double-check the Sheet ID from the URL
2. Verify the sheet is shared with the service account email
3. Make sure the service account has "Editor" access (not just "Viewer")

### If auto-connect fails

**Solution:**
1. Check your `.env` file has `GOOGLE_SHEET_ID=...`
2. Restart the Streamlit app
3. Or manually connect using the UI

---

## 📝 Quick Test

After setup, run a stock analysis and check:
1. You should see "✅ Synced to Google Sheets!" after analysis completes
2. Open your Google Sheet - you should see new data
3. The "Portfolio Analysis" worksheet will be auto-created

---

## 🚀 Status

- ✅ Libraries installed (`gspread`, `google-auth`)
- ✅ Credentials file detected (`google_credentials.json`)
- ✅ Streamlit app restarted

**Next Step:** Add your Sheet ID and share the sheet with your service account!

---

## 📧 Service Account Email

To find your service account email:
```bash
cat google_credentials.json | grep client_email
```

Or open `google_credentials.json` and look for the `client_email` field.
