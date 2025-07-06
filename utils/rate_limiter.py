"""
Rate limiting utility for API calls to respect service limits.
Specifically designed for Groq API free tier (30 requests/minute).
"""

import time
import logging
from threading import Lock
from collections import deque


class RateLimiter:
    """
    Rate limiter to control API request frequency.
    Thread-safe implementation using sliding window approach.
    """
    
    def __init__(self, max_requests: int = 30, time_window: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window.
            time_window: Time window in seconds (default: 60 seconds for 1 minute).
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)

    def wait_if_needed(self) -> None:
        """
        Wait if necessary to respect rate limits.
        Uses sliding window approach to track requests.
        """
        with self.lock:
            current_time = time.time()
            
            # Remove requests outside the time window
            while self.requests and current_time - self.requests[0] > self.time_window:
                self.requests.popleft()
            
            # Log current usage
            current_usage = len(self.requests)
            if current_usage > 0:
                self.logger.debug(f"API usage: {current_usage}/{self.max_requests} requests in last {self.time_window}s")
            
            # Check if we need to wait
            if current_usage >= self.max_requests:
                # Calculate wait time
                oldest_request = self.requests[0]
                wait_time = self.time_window - (current_time - oldest_request) + 0.5  # Add small buffer
                
                if wait_time > 0:
                    self.logger.info(f"Rate limit reached ({current_usage}/{self.max_requests}). Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    
                    # Clean up old requests after waiting
                    current_time = time.time()
                    while self.requests and current_time - self.requests[0] > self.time_window:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(current_time)
            
            # Log if we're approaching the limit
            new_usage = len(self.requests)
            if new_usage >= self.max_requests * 0.8:  # 80% threshold
                self.logger.warning(f"Approaching rate limit: {new_usage}/{self.max_requests} requests used")
            


# Global rate limiter instance for Groq API
groq_rate_limiter = RateLimiter(max_requests=25, time_window=60)  # Conservative limit

# Global rate limiter instance for Together API
together_rate_limiter = RateLimiter(max_requests=50, time_window=60)  # Together API limits
