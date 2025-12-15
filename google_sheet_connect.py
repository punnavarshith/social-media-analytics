"""
Google Sheets Connection Module
Handles authentication and connection to Google Sheets
"""

import gspread
from google.oauth2.service_account import Credentials
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def connect_to_google_sheets():
    """
    Connects to Google Sheets using service account credentials
    
    Returns:
        gspread.Client: Authenticated Google Sheets client
    """
    try:
        # Define the scope for Google Sheets and Google Drive API
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Authenticate using service account
        credentials = Credentials.from_service_account_file(
            'service_account.json',
            scopes=scopes
        )
        
        # Create client
        client = gspread.authorize(credentials)
        
        print("✅ Successfully connected to Google Sheets!")
        return client
        
    except FileNotFoundError:
        print("❌ Error: service_account.json not found!")
        print("Please download your service account key from Google Cloud Console")
        return None
    except Exception as e:
        print(f"❌ Error connecting to Google Sheets: {e}")
        return None


def get_sheet(client, sheet_id=None):
    """
    Gets a specific Google Sheet by ID
    
    Args:
        client: Authenticated gspread client
        sheet_id: Google Sheet ID (optional, uses env variable if not provided)
    
    Returns:
        gspread.Spreadsheet: The requested spreadsheet
    """
    try:
        if sheet_id is None:
            sheet_id = os.getenv('GOOGLE_SHEET_ID')
        
        if not sheet_id:
            print("❌ Error: No sheet ID provided!")
            return None
            
        spreadsheet = client.open_by_key(sheet_id)
        print(f"✅ Successfully opened spreadsheet: {spreadsheet.title}")
        return spreadsheet
        
    except Exception as e:
        print(f"❌ Error opening spreadsheet: {e}")
        return None


def create_worksheet(spreadsheet, title, rows=1000, cols=20):
    """
    Creates a new worksheet in the spreadsheet
    
    Args:
        spreadsheet: gspread.Spreadsheet object
        title: Name of the new worksheet
        rows: Number of rows (default 1000)
        cols: Number of columns (default 20)
    
    Returns:
        gspread.Worksheet: The created worksheet
    """
    try:
        worksheet = spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
        print(f"✅ Created new worksheet: {title}")
        return worksheet
    except Exception as e:
        print(f"⚠️ Worksheet '{title}' might already exist or error occurred: {e}")
        # Try to get existing worksheet
        try:
            worksheet = spreadsheet.worksheet(title)
            print(f"✅ Using existing worksheet: {title}")
            return worksheet
        except:
            return None


if __name__ == "__main__":
    # Test the connection
    print("Testing Google Sheets connection...")
    client = connect_to_google_sheets()
    
    if client:
        sheet = get_sheet(client)
        if sheet:
            print(f"📊 Spreadsheet Title: {sheet.title}")
            print(f"📊 Number of worksheets: {len(sheet.worksheets())}")
