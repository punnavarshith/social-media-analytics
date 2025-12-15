"""
Custom CSS styles for Streamlit app
Dark theme with clean, minimalist UI
"""

def get_custom_css():
    """Return custom CSS for the app"""
    return """
    <style>
        /* Main header styling */
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 20px 0;
            margin-bottom: 10px;
        }
        
        /* Page title styling */
        .page-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        /* Prediction box */
        .prediction-box {
            background: rgba(102, 126, 234, 0.1);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 15px 0;
        }
        
        /* Success box */
        .success-box {
            background: rgba(46, 204, 113, 0.1);
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #2ecc71;
            margin: 10px 0;
        }
        
        /* Warning box */
        .warning-box {
            background: rgba(241, 196, 15, 0.1);
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #f1c40f;
            margin: 10px 0;
        }
        
        /* Info box */
        .info-box {
            background: rgba(52, 152, 219, 0.1);
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #3498db;
            margin: 10px 0;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Card container */
        .card {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 15px 0;
        }
        
        /* Divider */
        .divider {
            height: 3px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            margin: 30px 0;
            border-radius: 2px;
        }
        
        /* Stats badge */
        .stats-badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 5px;
        }
        
        /* Section header */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #667eea;
            margin: 20px 0 10px 0;
            display: flex;
            align-items: center;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #1e1e1e 0%, #2d2d2d 100%);
        }
    </style>
    """
