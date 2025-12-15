"""
Google Trends Data Fetching Module
Fetches trending topics and search interest data using pytrends
"""

from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime, timedelta
import time


def authenticate_trends():
    """
    Creates a Google Trends request object (no authentication needed!)
    
    Returns:
        TrendReq: Google Trends request object
    """
    try:
        # Create pytrends object (no API key needed!)
        pytrends = TrendReq(hl='en-US', tz=360)
        print("✅ Successfully connected to Google Trends!")
        return pytrends
        
    except Exception as e:
        print(f"❌ Error connecting to Google Trends: {e}")
        return None


def get_trending_searches(pytrends, country='united_states'):
    """
    Fetches current trending searches for a specific country
    
    Args:
        pytrends: TrendReq object
        country: Country code (united_states, united_kingdom, india, etc.)
    
    Returns:
        pandas.DataFrame: DataFrame containing trending searches
    """
    try:
        print(f"🔍 Fetching trending searches for {country}...")
        
        # Get trending searches
        trending_df = pytrends.trending_searches(pn=country)
        
        # Rename column
        trending_df.columns = ['trending_query']
        
        # Add metadata
        trending_df['country'] = country
        trending_df['fetched_at'] = datetime.now()
        trending_df['rank'] = range(1, len(trending_df) + 1)
        
        print(f"✅ Fetched {len(trending_df)} trending searches!")
        return trending_df
        
    except Exception as e:
        print(f"❌ Error fetching trending searches: {e}")
        return pd.DataFrame()


def get_interest_over_time(pytrends, keywords, timeframe='today 3-m', geo=''):
    """
    Fetches search interest over time for specific keywords
    
    Args:
        pytrends: TrendReq object
        keywords: List of keywords (max 5)
        timeframe: Time range ('today 1-m', 'today 3-m', 'today 12-m', 'today 5-y', 'all')
        geo: Geographic location (e.g., 'US', 'GB', 'IN', '' for worldwide)
    
    Returns:
        pandas.DataFrame: DataFrame with search interest over time
    """
    try:
        if isinstance(keywords, str):
            keywords = [keywords]
        
        if len(keywords) > 5:
            print("⚠️ Warning: Maximum 5 keywords allowed. Using first 5.")
            keywords = keywords[:5]
        
        print(f"📊 Fetching interest over time for: {', '.join(keywords)}")
        print(f"   Timeframe: {timeframe}, Geography: {geo or 'Worldwide'}")
        
        # Build payload
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')
        
        # Get interest over time
        interest_df = pytrends.interest_over_time()
        
        if interest_df.empty:
            print("⚠️ No data found for these keywords!")
            return pd.DataFrame()
        
        # Remove 'isPartial' column if exists
        if 'isPartial' in interest_df.columns:
            interest_df = interest_df.drop(columns=['isPartial'])
        
        # Reset index to make 'date' a column
        interest_df = interest_df.reset_index()
        
        print(f"✅ Fetched interest data for {len(interest_df)} time periods!")
        return interest_df
        
    except Exception as e:
        print(f"❌ Error fetching interest over time: {e}")
        return pd.DataFrame()


def get_interest_by_region(pytrends, keywords, timeframe='today 3-m', resolution='COUNTRY'):
    """
    Fetches search interest by geographic region
    
    Args:
        pytrends: TrendReq object
        keywords: List of keywords (max 5)
        timeframe: Time range
        resolution: Geographic resolution ('COUNTRY', 'REGION', 'CITY', 'DMA')
    
    Returns:
        pandas.DataFrame: DataFrame with interest by region
    """
    try:
        if isinstance(keywords, str):
            keywords = [keywords]
        
        if len(keywords) > 5:
            keywords = keywords[:5]
        
        print(f"🌍 Fetching interest by region for: {', '.join(keywords)}")
        
        # Build payload
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
        
        # Get interest by region
        region_df = pytrends.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=True)
        
        if region_df.empty:
            print("⚠️ No regional data found!")
            return pd.DataFrame()
        
        # Reset index to make region a column
        region_df = region_df.reset_index()
        region_df = region_df.rename(columns={'geoName': 'region'})
        
        print(f"✅ Fetched regional data for {len(region_df)} regions!")
        return region_df
        
    except Exception as e:
        print(f"❌ Error fetching interest by region: {e}")
        return pd.DataFrame()


def get_related_queries(pytrends, keywords, timeframe='today 3-m'):
    """
    Fetches related queries (top and rising) for keywords
    
    Args:
        pytrends: TrendReq object
        keywords: List of keywords (max 5)
        timeframe: Time range
    
    Returns:
        dict: Dictionary with 'top' and 'rising' DataFrames
    """
    try:
        if isinstance(keywords, str):
            keywords = [keywords]
        
        if len(keywords) > 5:
            keywords = keywords[:5]
        
        print(f"🔗 Fetching related queries for: {', '.join(keywords)}")
        
        # Build payload
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
        
        # Get related queries
        related_dict = pytrends.related_queries()
        
        # Process results
        all_top = []
        all_rising = []
        
        for keyword in keywords:
            if keyword in related_dict:
                # Top queries
                if related_dict[keyword]['top'] is not None:
                    top_df = related_dict[keyword]['top'].copy()
                    top_df['keyword'] = keyword
                    top_df['type'] = 'top'
                    all_top.append(top_df)
                
                # Rising queries
                if related_dict[keyword]['rising'] is not None:
                    rising_df = related_dict[keyword]['rising'].copy()
                    rising_df['keyword'] = keyword
                    rising_df['type'] = 'rising'
                    all_rising.append(rising_df)
        
        result = {}
        
        if all_top:
            result['top'] = pd.concat(all_top, ignore_index=True)
            print(f"✅ Found {len(result['top'])} top related queries!")
        else:
            result['top'] = pd.DataFrame()
            print("⚠️ No top related queries found")
        
        if all_rising:
            result['rising'] = pd.concat(all_rising, ignore_index=True)
            print(f"✅ Found {len(result['rising'])} rising related queries!")
        else:
            result['rising'] = pd.DataFrame()
            print("⚠️ No rising related queries found")
        
        return result
        
    except Exception as e:
        print(f"❌ Error fetching related queries: {e}")
        return {'top': pd.DataFrame(), 'rising': pd.DataFrame()}


def get_related_topics(pytrends, keywords, timeframe='today 3-m'):
    """
    Fetches related topics for keywords
    
    Args:
        pytrends: TrendReq object
        keywords: List of keywords (max 5)
        timeframe: Time range
    
    Returns:
        dict: Dictionary with 'top' and 'rising' DataFrames
    """
    try:
        if isinstance(keywords, str):
            keywords = [keywords]
        
        if len(keywords) > 5:
            keywords = keywords[:5]
        
        print(f"📑 Fetching related topics for: {', '.join(keywords)}")
        
        # Build payload
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
        
        # Get related topics
        related_dict = pytrends.related_topics()
        
        # Process results
        all_top = []
        all_rising = []
        
        for keyword in keywords:
            if keyword in related_dict:
                # Top topics
                if related_dict[keyword]['top'] is not None:
                    top_df = related_dict[keyword]['top'].copy()
                    top_df['keyword'] = keyword
                    top_df['type'] = 'top'
                    all_top.append(top_df)
                
                # Rising topics
                if related_dict[keyword]['rising'] is not None:
                    rising_df = related_dict[keyword]['rising'].copy()
                    rising_df['keyword'] = keyword
                    rising_df['type'] = 'rising'
                    all_rising.append(rising_df)
        
        result = {}
        
        if all_top:
            result['top'] = pd.concat(all_top, ignore_index=True)
            print(f"✅ Found {len(result['top'])} top related topics!")
        else:
            result['top'] = pd.DataFrame()
        
        if all_rising:
            result['rising'] = pd.concat(all_rising, ignore_index=True)
            print(f"✅ Found {len(result['rising'])} rising related topics!")
        else:
            result['rising'] = pd.DataFrame()
        
        return result
        
    except Exception as e:
        print(f"❌ Error fetching related topics: {e}")
        return {'top': pd.DataFrame(), 'rising': pd.DataFrame()}


if __name__ == "__main__":
    # Test Google Trends API
    print("Testing Google Trends API...")
    print("="*60)
    
    pytrends = authenticate_trends()
    
    if pytrends:
        # Test 1: Get trending searches
        print("\n" + "="*60)
        print("TEST 1: Trending Searches (USA)")
        print("="*60)
        trending_df = get_trending_searches(pytrends, country='united_states')
        if not trending_df.empty:
            print("\n🔥 Top 10 Trending Searches:")
            print(trending_df[['rank', 'trending_query']].head(10))
        
        # Add delay to avoid rate limiting
        time.sleep(2)
        
        # Test 2: Interest over time
        print("\n" + "="*60)
        print("TEST 2: Interest Over Time")
        print("="*60)
        keywords = ['artificial intelligence', 'machine learning']
        interest_df = get_interest_over_time(pytrends, keywords, timeframe='today 3-m')
        if not interest_df.empty:
            print("\n📊 Sample interest data:")
            print(interest_df.tail(5))
        
        # Add delay
        time.sleep(2)
        
        # Test 3: Related queries
        print("\n" + "="*60)
        print("TEST 3: Related Queries")
        print("="*60)
        related = get_related_queries(pytrends, ['content marketing'], timeframe='today 3-m')
        if not related['rising'].empty:
            print("\n🚀 Rising Related Queries:")
            print(related['rising'].head(5))
