"""
Twitter Multi-Account Manager
Rotates through multiple Twitter accounts to avoid rate limits
"""

import tweepy
import json
import time
from datetime import datetime


class TwitterAccountManager:
    """
    Manages multiple Twitter API accounts for rate limit avoidance
    """
    
    def __init__(self, accounts_file='twitter_accounts.json', skip_accounts=None):
        """
        Initialize with accounts from JSON file
        
        Args:
            accounts_file: Path to JSON file with account credentials
            skip_accounts: List of account names to skip (e.g., ['Account_1'])
        """
        self.accounts_file = accounts_file
        self.accounts = []
        self.clients = []
        self.current_index = 0
        self.skip_accounts = skip_accounts or []
        self.load_accounts()
    
    def load_accounts(self):
        """Load Twitter accounts from JSON file"""
        try:
            with open(self.accounts_file, 'r') as f:
                all_accounts = json.load(f)
            
            # Filter out skipped accounts
            if self.skip_accounts:
                print(f"⚠️ Skipping accounts: {', '.join(self.skip_accounts)}")
                self.accounts = [acc for acc in all_accounts if acc['name'] not in self.skip_accounts]
            else:
                self.accounts = all_accounts
            
            print(f"✅ Loaded {len(self.accounts)} Twitter accounts")
            
            # Create clients for all accounts
            for i, account in enumerate(self.accounts, 1):
                try:
                    client = tweepy.Client(
                        bearer_token=account['bearer_token'],
                        consumer_key=account['api_key'],
                        consumer_secret=account['api_secret'],
                        access_token=account['access_token'],
                        access_token_secret=account['access_secret'],
                        wait_on_rate_limit=False  # We'll handle rate limits manually
                    )
                    self.clients.append(client)
                    print(f"   ✓ Account {i} ({account['name']}): Ready")
                except Exception as e:
                    print(f"   ✗ Account {i} ({account['name']}): Failed - {e}")
            
            if not self.clients:
                raise Exception("No valid Twitter accounts loaded!")
            
            print(f"✅ {len(self.clients)} accounts ready to use!\n")
            
        except FileNotFoundError:
            print(f"❌ File not found: {self.accounts_file}")
            print("Please create twitter_accounts.json with your API keys")
            raise
        except Exception as e:
            print(f"❌ Error loading accounts: {e}")
            raise
    
    def get_current_client(self):
        """Get the current active Twitter client"""
        if not self.clients:
            raise Exception("No Twitter clients available!")
        return self.clients[self.current_index]
    
    def get_current_account_name(self):
        """Get the name of current account"""
        return self.accounts[self.current_index]['name']
    
    def rotate_account(self):
        """Switch to next account"""
        self.current_index = (self.current_index + 1) % len(self.clients)
        print(f"🔄 Switched to account: {self.get_current_account_name()}")
        return self.get_current_client()
    
    def fetch_with_rotation(self, fetch_function, *args, max_retries=None, **kwargs):
        """
        Execute a fetch function with automatic account rotation on rate limits
        
        Args:
            fetch_function: Function to call (e.g., search_tweets)
            max_retries: Maximum number of accounts to try (default: all accounts)
            *args, **kwargs: Arguments to pass to fetch_function
        
        Returns:
            Result from fetch_function or None if all accounts exhausted
        """
        if max_retries is None:
            max_retries = len(self.clients)
        
        attempts = 0
        
        while attempts < max_retries:
            client = self.get_current_client()
            account_name = self.get_current_account_name()
            
            try:
                # Try to fetch with current account
                result = fetch_function(client, *args, **kwargs)
                return result
                
            except tweepy.errors.TooManyRequests as e:
                print(f"   ⚠️ Rate limit hit on {account_name}")
                attempts += 1
                
                if attempts < max_retries:
                    # Rotate to next account
                    self.rotate_account()
                    time.sleep(1)  # Brief pause before retry
                else:
                    print(f"   ❌ All {max_retries} accounts rate limited!")
                    return None
                    
            except Exception as e:
                print(f"   ❌ Error with {account_name}: {e}")
                attempts += 1
                
                if attempts < max_retries:
                    self.rotate_account()
                    time.sleep(1)
                else:
                    return None
        
        return None


def authenticate_twitter_multi(skip_accounts=None):
    """
    Create TwitterAccountManager instance
    
    Args:
        skip_accounts: List of account names to skip (e.g., ['Account_1'])
    
    Returns:
        TwitterAccountManager: Manager with multiple accounts
    """
    try:
        manager = TwitterAccountManager(skip_accounts=skip_accounts)
        return manager
    except Exception as e:
        print(f"❌ Failed to initialize Twitter account manager: {e}")
        return None


if __name__ == "__main__":
    # Test the multi-account system
    print("Testing Twitter Multi-Account Manager")
    print("=" * 60)
    
    manager = authenticate_twitter_multi()
    
    if manager:
        print(f"\n✅ Successfully loaded {len(manager.clients)} accounts")
        print(f"Current account: {manager.get_current_account_name()}")
        
        print("\nTesting account rotation:")
        for i in range(3):
            print(f"  {i+1}. Current: {manager.get_current_account_name()}")
            manager.rotate_account()
