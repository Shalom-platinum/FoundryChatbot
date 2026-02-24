"""
Agentic tools for extending AI capabilities
"""
import logging
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

# Use DuckDuckGo for search
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


class WebSearchAgent:
    """Agent for performing web searches"""
    
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Search the web using DuckDuckGo and fetch full content for top results
        
        Args:
            query: Search query
            max_results: Maximum number of results to fetch full content for
            
        Returns:
            Dict with search results and metadata
        """
        try:
            logger.info(f"Starting DDGS search for query: {query}")
            # Get basic search results
            basic_results = list(self.ddgs.text(query, max_results=max_results * 2))
            logger.info(f"DDGS search returned {len(basic_results)} results")
            
            formatted_results = []
            for result in basic_results:
                try:
                    url = result.get('href', result.get('link', ''))
                    title = result.get('title', '')
                    snippet = result.get('body', result.get('snippet', ''))
                    logger.info(f"Processing result: {title[:50]}... URL: {url}")
                    
                    # Skip obviously irrelevant results
                    if self._is_irrelevant_result(url, title, snippet):
                        logger.info(f"Skipping irrelevant result: {url}")
                        continue
                    
                    # Fetch full content for more context
                    full_content = self._fetch_page_content(url)
                    logger.info(f"Fetched content length: {len(full_content) if full_content else 0}")
                    
                    # Check if content is relevant to the query
                    if not self._is_content_relevant(full_content, query):
                        logger.info(f"Content not relevant to query, skipping")
                        continue
                    
                    # Only include if we got meaningful content
                    if full_content and len(full_content) > 100:
                        formatted_results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'full_content': full_content
                        })
                        logger.info(f"Added result, total now: {len(formatted_results)}")
                    
                    if len(formatted_results) >= max_results:
                        break
                except Exception as e:
                    logger.error(f"Error processing result {result}: {e}")
                    continue
            
            logger.info(f"Returning {len(formatted_results)} formatted results")
            return {
                'success': True,
                'query': query,
                'results': formatted_results,
                'count': len(formatted_results)
            }
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'results': []
            }
    
    def _is_irrelevant_result(self, url: str, title: str, snippet: str) -> bool:
        """Check if a search result appears to be irrelevant"""
        url_lower = url.lower()
        title_lower = title.lower()
        snippet_lower = snippet.lower()
        
        # Skip download sites
        download_indicators = ['download', 'free download', 'installer', 'setup.exe', 'softonic', 'cnet']
        if any(indicator in url_lower for indicator in download_indicators):
            return True
        
        # Skip obvious ad/tracker sites
        ad_sites = ['doubleclick', 'googlesyndication', 'amazon-adsystem', 'facebook.com/l.php']
        if any(site in url_lower for site in ad_sites):
            return True
        
        # Skip if title/snippet is too short or generic
        if len(title + snippet) < 20:
            return True
        
        # Skip if it looks like a redirect or tracking URL
        if 'utm_' in url_lower or 'redirect' in url_lower:
            return True
        
        return False

    def _is_content_relevant(self, content: str, query: str) -> bool:
        """Check if the fetched content is relevant to the query"""
        if not content:
            return False
        
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Extract key terms from query (words longer than 3 chars, excluding common words)
        common_words = {'what', 'is', 'the', 'latest', 'between', 'vs', 'on', 'use', 'websites', 'like', 'or', 'and', 'for', 'are', 'how', 'why', 'when', 'where'}
        key_terms = [word for word in query_lower.split() if len(word) > 3 and word not in common_words]
        
        # Check if at least one key term appears in the content
        relevant_terms_found = any(term in content_lower for term in key_terms)
        
        logger.info(f"Key terms: {key_terms}, Relevant terms found: {relevant_terms_found}")
        return relevant_terms_found

    def _fetch_page_content(self, url: str, max_length: int = 2000) -> str:
        """
        Fetch and extract main content from a webpage
        
        Args:
            url: URL to fetch
            max_length: Maximum content length to return
            
        Returns:
            Extracted text content
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'main', 'article', '.content', '.post-content', 
                '.entry-content', '.article-content', '#content',
                '.story-body', '.article-body'
            ]
            
            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            # Fallback to body if no specific content area found
            if not content:
                content = soup.body or soup
            
            # Extract text
            text = content.get_text(separator=' ', strip=True)
            
            # Limit length
            if len(text) > max_length:
                text = text[:max_length] + "..."
                print("Truncated fetched content to fit max length", text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return f"Could not fetch content: {str(e)}"

    def format_for_context(self, search_results: Dict[str, Any]) -> str:
        """Format search results as context for AI"""
        if not search_results.get('success'):
            return f"Web search failed: {search_results.get('error', 'Unknown error')}"
        
        context = f"Web search results for '{search_results['query']}':\n\n"
        for i, result in enumerate(search_results['results'], 1):
            context += f"{i}. **{result['title']}**\n"
            context += f"   URL: {result['url']}\n"
            context += f"   Snippet: {result['snippet']}\n"
            if result.get('full_content'):
                context += f"   Full Content: {result['full_content']}\n"
            context += "\n"
        
        return context


class CodeExecutionAgent:
    """Agent for executing Python code safely"""
    
    TIMEOUT_SECONDS = 10
    MAX_OUTPUT_LENGTH = 5000
    
    # Restricted imports for safety
    RESTRICTED_IMPORTS = [
        'os.system', 'subprocess', 'shutil.rmtree', 
        'eval', 'exec', 'compile', '__import__'
    ]
    
    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in a restricted environment
        
        Args:
            code: Python code to execute
            
        Returns:
            Dict with execution results
        """
        # Basic security check
        for restricted in self.RESTRICTED_IMPORTS:
            if restricted in code:
                return {
                    'success': False,
                    'error': f"Restricted operation detected: {restricted}",
                    'output': '',
                    'code': code
                }
        
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute with timeout
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.TIMEOUT_SECONDS
                )
                
                output = result.stdout
                error = result.stderr
                
                # Truncate if too long
                if len(output) > self.MAX_OUTPUT_LENGTH:
                    output = output[:self.MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
                if len(error) > self.MAX_OUTPUT_LENGTH:
                    error = error[:self.MAX_OUTPUT_LENGTH] + "\n... (error truncated)"
                
                return {
                    'success': result.returncode == 0,
                    'output': output,
                    'error': error,
                    'return_code': result.returncode,
                    'code': code
                }
                
            finally:
                # Clean up temp file
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Code execution timed out after {self.TIMEOUT_SECONDS} seconds",
                'output': '',
                'code': code
            }
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'output': '',
                'code': code
            }

    def format_for_context(self, execution_result: Dict[str, Any]) -> str:
        """Format execution results as context for AI"""
        context = "Code Execution Result:\n"
        context += f"```python\n{execution_result['code']}\n```\n\n"
        
        if execution_result['success']:
            context += "**Output:**\n"
            context += f"```\n{execution_result['output']}\n```"
        else:
            context += f"**Error:**\n```\n{execution_result.get('error', 'Unknown error')}\n```"
        
        return context


class AgentOrchestrator:
    """Orchestrates multiple agents based on user intent"""
    
    def __init__(self):
        self.web_search = WebSearchAgent()
        self.code_execution = CodeExecutionAgent()
    
    def detect_intent(self, message: str) -> Dict[str, bool]:
        """
        Detect user intent from message
        
        Args:
            message: User message
            
        Returns:
            Dict with detected intents
        """
        message_lower = message.lower()
        
        return {
            'web_search': any(phrase in message_lower for phrase in [
                'search the web', 'look up', 'find online', 'search for',
                'what is the latest', 'recent news', 'current', 'today'
            ]),
            'code_execution': any(phrase in message_lower for phrase in [
                'run this code', 'execute', 'try this python', 'test this code',
                '```python'
            ])
        }
    
    def extract_code(self, message: str) -> Optional[str]:
        """Extract Python code from message"""
        import re
        # Look for code blocks
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, message, re.DOTALL)
        if matches:
            return matches[0]
        return None
    
    def process_with_tools(
        self, 
        message: str, 
        enable_web_search: bool = False,
        enable_code_execution: bool = False
    ) -> Dict[str, Any]:
        """
        Process message with available tools
        
        Args:
            message: User message
            enable_web_search: Whether web search is enabled
            enable_code_execution: Whether code execution is enabled
            
        Returns:
            Dict with tool results and context
        """
        results = {
            'tool_calls': [],
            'tool_results': [],
            'context': ''
        }
        
        intents = self.detect_intent(message)
        
        # Web search
        if enable_web_search and intents['web_search']:
            print("using web tools under agent file")
            # Extract search query (simple approach)
            query = message
            for phrase in ['search the web for', 'look up', 'search for']:
                if phrase in message.lower():
                    idx = message.lower().index(phrase)
                    query = message[idx + len(phrase):].strip()
                    break
            
            search_results = self.web_search.search(query)
            results['tool_calls'].append({
                'tool': 'web_search',
                'input': query
            })
            results['tool_results'].append(search_results)
            results['context'] += self.web_search.format_for_context(search_results) + "\n\n"
        
        # Code execution
        if enable_code_execution and intents['code_execution']:
            code = self.extract_code(message)
            if code:
                exec_result = self.code_execution.execute(code)
                results['tool_calls'].append({
                    'tool': 'code_execution',
                    'input': code
                })
                results['tool_results'].append(exec_result)
                results['context'] += self.code_execution.format_for_context(exec_result) + "\n\n"
        print("results from agent file", results)
        return results


# Global instance
agent_orchestrator = AgentOrchestrator()
