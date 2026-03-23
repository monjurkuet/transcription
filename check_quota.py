#!/usr/bin/env python3
"""
Groq API Quota Checker CLI

Check remaining quota for configured Groq API keys.
Supports text and JSON output formats.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv

from groq_key_manager import RoundRobinKeyManager


def format_timedelta(dt: Optional[datetime]) -> str:
    """Format a datetime as a human-readable 'in Xh Ym Zs' string."""
    if dt is None:
        return "N/A"
    
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    diff = dt - now
    
    if diff.total_seconds() <= 0:
        return "now"
    
    total_seconds = int(diff.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)


def check_key_quota_via_api(key: str) -> Dict[str, Any]:
    """
    Check quota for a single key by making an API call.
    
    Args:
        key: The API key to check
        
    Returns:
        Dictionary with quota information from response headers
    """
    result = {
        "key": key[:8] + "***" + key[-6:] if len(key) > 12 else key[:4] + "***",
        "requests_remaining": None,
        "requests_limit": None,
        "tokens_remaining": None,
        "tokens_limit": None,
        "reset_requests": None,
        "reset_tokens": None,
        "error": None,
    }
    
    try:
        # Use chat completions endpoint - it's lightweight and returns rate limit headers
        # This works for checking general API quota (requests per day)
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1
            },
            timeout=10
        )
        
        headers = dict(response.headers)
        
        # Parse rate limit headers
        if "x-ratelimit-remaining-requests" in headers:
            try:
                result["requests_remaining"] = int(headers["x-ratelimit-remaining-requests"])
            except (ValueError, TypeError):
                pass
        
        if "x-ratelimit-limit-requests" in headers:
            try:
                result["requests_limit"] = int(headers["x-ratelimit-limit-requests"])
            except (ValueError, TypeError):
                pass
        
        if "x-ratelimit-remaining-tokens" in headers:
            try:
                result["tokens_remaining"] = int(headers["x-ratelimit-remaining-tokens"])
            except (ValueError, TypeError):
                pass
        
        if "x-ratelimit-limit-tokens" in headers:
            try:
                result["tokens_limit"] = int(headers["x-ratelimit-limit-tokens"])
            except (ValueError, TypeError):
                pass
        
        if "x-ratelimit-reset-requests" in headers:
            result["reset_requests"] = _parse_reset_time(headers["x-ratelimit-reset-requests"])
        
        if "x-ratelimit-reset-tokens" in headers:
            result["reset_tokens"] = _parse_reset_time(headers["x-ratelimit-reset-tokens"])
        
        if not response.ok:
            result["error"] = f"API error: {response.status_code}"
        
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out"
    except requests.exceptions.RequestException as e:
        result["error"] = f"Request failed: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


def _parse_reset_time(reset_str: str) -> Optional[str]:
    """
    Parse rate limit reset time string to human-readable format.
    
    Handles formats like "2h59.56s" (hours, minutes, seconds).
    """
    if not reset_str:
        return None
    
    try:
        reset_str = reset_str.strip()
        hours = 0
        minutes = 0
        seconds = 0.0
        
        if "h" in reset_str:
            parts = reset_str.split("h")
            hours = int(parts[0])
            reset_str = parts[1] if len(parts) > 1 else "0s"
        
        if "m" in reset_str:
            parts = reset_str.split("m")
            minutes = int(parts[0])
            reset_str = parts[1] if len(parts) > 1 else "0s"
        
        if "s" in reset_str:
            seconds = float(reset_str.replace("s", ""))
        
        total_seconds = hours * 3600 + minutes * 60 + seconds
        
        # Return both absolute and human-readable
        abs_time = datetime.now(timezone.utc).replace(microsecond=0)
        abs_time = abs_time.__add__(__import__("datetime").timedelta(seconds=total_seconds))
        
        return {
            "absolute": abs_time.isoformat(),
            "relative": format_timedelta(abs_time),
        }
    except (ValueError, AttributeError):
        return None


def format_quota_text(results: list, checked_at: str) -> str:
    """Format quota results as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("Groq API Quota Status")
    lines.append("=" * 60)
    lines.append(f"Checked at: {checked_at}")
    lines.append(f"Total keys: {len(results)}")
    lines.append("")
    
    for i, result in enumerate(results, 1):
        lines.append(f"Key {i}: {result['key']}")
        
        if result.get("error"):
            lines.append(f"  Status: ERROR - {result['error']}")
        else:
            # Requests
            if result.get("requests_limit") is not None:
                remaining = result.get("requests_remaining", "N/A")
                limit = result["requests_limit"]
                pct = (remaining / limit * 100) if isinstance(remaining, (int, float)) and remaining is not None else 0
                lines.append(f"  Requests: {remaining} / {limit} (daily limit) [{pct:.1f}%]")
                
                reset = result.get("reset_requests")
                if reset:
                    if isinstance(reset, dict):
                        lines.append(f"  Resets: in {reset.get('relative', 'N/A')}")
                    else:
                        lines.append(f"  Resets: {reset}")
            else:
                lines.append("  Requests: N/A")
            
            # Tokens
            if result.get("tokens_limit") is not None:
                remaining = result.get("tokens_remaining", "N/A")
                limit = result["tokens_limit"]
                lines.append(f"  Tokens: {remaining} / {limit} (per minute)")
                
                reset = result.get("reset_tokens")
                if reset:
                    if isinstance(reset, dict):
                        lines.append(f"  Token reset: in {reset.get('relative', 'N/A')}")
                    else:
                        lines.append(f"  Token reset: {reset}")
            else:
                lines.append("  Tokens: N/A (audio model)")
        
        lines.append("")
    
    return "\n".join(lines)


def format_quota_json(results: list, checked_at: str) -> str:
    """Format quota results as JSON."""
    output = {
        "checked_at": checked_at,
        "num_keys": len(results),
        "keys": []
    }
    
    for result in results:
        key_info = {
            "key": result["key"],
            "error": result.get("error"),
            "requests": None,
            "tokens": None,
        }
        
        if not result.get("error"):
            if result.get("requests_limit") is not None:
                key_info["requests"] = {
                    "remaining": result.get("requests_remaining"),
                    "limit": result["requests_limit"],
                    "reset": result.get("reset_requests"),
                }
            
            if result.get("tokens_limit") is not None:
                key_info["tokens"] = {
                    "remaining": result.get("tokens_remaining"),
                    "limit": result["tokens_limit"],
                    "reset": result.get("reset_tokens"),
                }
        
        output["keys"].append(key_info)
    
    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Check Groq API quota for configured keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Check quota for all keys (text output)
  %(prog)s --json              Check quota with JSON output
  %(prog)s --list-keys          List configured keys without checking
  %(prog)s --check-headers      Check quota via API response headers
        """
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format"
    )
    
    parser.add_argument(
        "--list-keys", "-l",
        action="store_true",
        help="List configured API keys (masked)"
    )
    
    parser.add_argument(
        "--check-headers", "-c",
        action="store_true",
        help="Check quota via API response headers (makes a request)"
    )
    
    args = parser.parse_args()
    
    # Load .env
    load_dotenv()
    
    try:
        manager = RoundRobinKeyManager()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please set GROQ_API_KEYS or GROQ_API_KEY in your .env file", file=sys.stderr)
        sys.exit(1)
    
    checked_at = datetime.now(timezone.utc).isoformat()
    
    # List keys mode
    if args.list_keys:
        quota_info = manager.get_quota_info()
        
        if args.json:
            output = {
                "num_keys": quota_info["num_keys"],
                "keys": [
                    {"key": k["key"], "is_exhausted": k["is_exhausted"]}
                    for k in quota_info["keys"]
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Configured API Keys ({quota_info['num_keys']} total)")
            print("=" * 50)
            for i, k in enumerate(quota_info["keys"], 1):
                status = "EXHAUSTED" if k["is_exhausted"] else "available"
                print(f"  {i}. {k['key']} [{status}]")
        
        sys.exit(0)
    
    # Check quota
    results = []
    
    if args.check_headers:
        # Make API calls to get rate limit headers
        for key in manager.keys:
            result = check_key_quota_via_api(key)
            results.append(result)
    else:
        # Just show tracked state
        quota_info = manager.get_quota_info()
        for k in quota_info["keys"]:
            results.append({
                "key": k["key"],
                "requests_remaining": k["requests_remaining"],
                "requests_limit": k["requests_limit"],
                "tokens_remaining": k["tokens_remaining"],
                "tokens_limit": k["tokens_limit"],
                "reset_requests": k["reset_requests"],
                "reset_tokens": k["reset_tokens"],
                "error": None,
            })
    
    # Format output
    if args.json:
        print(format_quota_json(results, checked_at))
    else:
        print(format_quota_text(results, checked_at))


if __name__ == "__main__":
    main()
