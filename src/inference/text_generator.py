"""
Inference engine for AneryzAI LLM
Supports various sampling strategies and multi-language generation
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import requests
from bs4 import BeautifulSoup
import re

class GenerationType(Enum):
    """Generation strategy type"""
    GREEDY = "greedy"
    SAMPLING = "sampling"
    TOP_K = "top_k"
    TOP_P = "top_p"
    BEAM_SEARCH = "beam_search"


class TextGenerator:
    """
    Text generator dengan berbagai sampling strategies
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def _prepare_input(
        self,
        text: str,
        language: str = "en",
        add_special_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Prepare input text for generation
        Args:
            text: Input text
            language: Language code ("en" or "id")
            add_special_tokens: Whether to add special tokens
        
        Returns:
            Token IDs tensor of shape (1, seq_len)
        """
        token_ids = self.tokenizer.encode(text, language=language, add_special_tokens=add_special_tokens)
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)
    
    def generate(
        self,
        prompt: str,
        language: str = "en",
        max_new_tokens: int = 100,
        generation_type: GenerationType = GenerationType.TOP_K,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        num_beams: int = 1,
        repetition_penalty: float = 1.2,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        return_scores: bool = False,
    ) -> Tuple[str, Optional[float]]:
        """
        Generate text completion given a prompt
        
        Args:
            prompt: Input prompt text
            language: Language code
            max_new_tokens: Maximum number of tokens to generate
            generation_type: Generation strategy
            temperature: Softmax temperature
            top_k: Top-K sampling parameter
            top_p: Top-P (nucleus) sampling parameter
            num_beams: Number of beams (1=greedy/sampling)
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Length penalty for longer sequences
            early_stopping: Stop generation when all beams finish
            return_scores: Whether to return generation score/probability
        
        Returns:
            Tuple of (generated_text, score)
        """
        with torch.no_grad():
            input_ids = self._prepare_input(prompt, language)
            
            if generation_type == GenerationType.BEAM_SEARCH:
                generated_ids = self._beam_search(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                    repetition_penalty=repetition_penalty,
                )
            else:
                # Generate menggunakan model's built-in generate method atau custom sampling
                generated_ids = self._sample_generation(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    generation_type=generation_type,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            
            # Decode generated tokens
            generated_text = self.tokenizer.decode(generated_ids[0].cpu().tolist())
            
            # Calculate score if requested
            score = None
            if return_scores:
                score = self._calculate_score(generated_ids)
            
            return generated_text, score
    
    def _calculate_score(self, generated_ids: torch.Tensor) -> float:
        """
        Calculate generation score/probability
        
        Args:
            generated_ids: Generated token IDs
        
        Returns:
            Average log probability score
        """
        with torch.no_grad():
            output = self.model(generated_ids)
            logits = output["logits"]
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = generated_ids[..., 1:].contiguous()
            
            # Calculate log probabilities
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Average log probability
            score = token_log_probs.mean().item()
            
            return score
    
    def think_and_generate(
        self,
        prompt: str,
        language: str = "en",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        num_steps: int = 3,
    ) -> str:
        """
        Generate text with chain-of-thought reasoning
        
        Args:
            prompt: Input prompt
            language: Language code ("en" or "id")
            max_new_tokens: Maximum tokens per step
            temperature: Sampling temperature
            top_k: Top-K sampling
            top_p: Top-P sampling
            num_steps: Number of reasoning steps
        
        Returns:
            Generated text with reasoning steps
        """
        # Create chain-of-thought prompt
        if language == "en":
            cot_prompt = f"Let's think step by step about: {prompt}\n\nStep 1:"
        else:
            cot_prompt = f"Mari kita pikirkan langkah demi langkah tentang: {prompt}\n\nLangkah 1:"
        
        full_response = cot_prompt
        
        for step in range(1, num_steps + 1):
            # Generate reasoning for this step
            step_text, _ = self.generate(
                full_response,
                language=language,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generation_type=GenerationType.TOP_P,
            )
            
            # Extract the new reasoning
            if language == "en":
                next_step_marker = f"Step {step + 1}:" if step < num_steps else "Final answer:"
            else:
                next_step_marker = f"Langkah {step + 1}:" if step < num_steps else "Jawaban akhir:"
            
            # Find where the next step begins
            marker_pos = step_text.find(next_step_marker)
            if marker_pos != -1:
                reasoning = step_text[len(full_response):marker_pos].strip()
                full_response += reasoning + "\n\n" + next_step_marker
            else:
                # If no next step marker, take all new text
                reasoning = step_text[len(full_response):].strip()
                full_response += reasoning
                break
        
        return full_response
    
    def search_and_generate(
        self,
        query: str,
        language: str = "en",
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """
        Search for information and generate response
        
        Args:
            query: Search query
            language: Language code
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-K sampling
            top_p: Top-P sampling
        
        Returns:
            Generated response with search context
        """
        # Perform web search (using DuckDuckGo instant answers)
        search_results = self._web_search(query, language)
        
        # Create prompt with search context
        if language == "en":
            context_prompt = f"Based on the following search results, answer the question: {query}\n\nSearch Results:\n{search_results}\n\nAnswer:"
        else:
            context_prompt = f"Berdasarkan hasil pencarian berikut, jawab pertanyaan: {query}\n\nHasil Pencarian:\n{search_results}\n\nJawaban:"
        
        # Generate response
        response, _ = self.generate(
            context_prompt,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generation_type=GenerationType.TOP_P,
        )
        
        return response
    
    def _web_search(self, query: str, language: str = "en") -> str:
        """
        Perform web search using DuckDuckGo instant answers
        
        Args:
            query: Search query
            language: Language code
        
        Returns:
            Search results as formatted text
        """
        try:
            # Use DuckDuckGo instant answers API
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            if language == "id":
                url += "&kl=id-id"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            results = []
            
            # Add instant answer if available
            if data.get('Answer'):
                results.append(f"Instant Answer: {data['Answer']}")
            
            # Add abstract if available
            if data.get('AbstractText'):
                results.append(f"Abstract: {data['AbstractText']}")
            
            # Add related topics
            if data.get('RelatedTopics'):
                for topic in data['RelatedTopics'][:3]:  # Limit to 3
                    if topic.get('Text'):
                        results.append(f"Related: {topic['Text']}")
            
            if not results:
                return "No search results found."
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def _sample_generation(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        generation_type: GenerationType,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """
        Autoregressive sampling with various strategies
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                output = self.model(
                    generated_ids,
                    use_cache=False,
                )
                
                logits = output["logits"][:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply repetition penalty
            for i in range(batch_size):
                for token_id in set(generated_ids[i].cpu().tolist()):
                    logits[i, token_id] /= repetition_penalty
            
            # Select next token based on generation type
            if generation_type == GenerationType.GREEDY:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            elif generation_type == GenerationType.SAMPLING:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            elif generation_type == GenerationType.TOP_K:
                next_token = self._top_k_sampling(logits, top_k)
            
            elif generation_type == GenerationType.TOP_P:
                next_token = self._top_p_sampling(logits, top_p)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        return generated_ids
    
    @staticmethod
    def _top_k_sampling(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Top-K sampling"""
        batch_size = logits.shape[0]
        
        # Get top-k logits
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        
        # Mask out tokens below top-k
        logits_mask = torch.full_like(logits, float('-inf'))
        logits_mask.scatter_(1, top_k_indices, top_k_logits)
        
        # Sample from filtered distribution
        probs = torch.softmax(logits_mask, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
    
    @staticmethod
    def _top_p_sampling(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Top-P (nucleus) sampling"""
        # Sort logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Compute cumulative probabilities
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff index
        cutoff_mask = cumsum_probs <= top_p
        cutoff_mask[:, 0] = True  # Keep at least one token
        
        # Mask out tokens beyond cutoff
        logits_mask = logits.clone()
        logits_mask[~cutoff_mask] = float('-inf')
        
        # Sample from filtered distribution
        probs = torch.softmax(logits_mask, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
    
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        length_penalty: float,
        early_stopping: bool,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """
        Beam search dengan length normalization
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize beams
        # Shape: (batch_size * num_beams, seq_len)
        input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
        input_ids = input_ids.view(batch_size * num_beams, -1)
        
        # Track beam scores and finished beams
        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)
        
        past_key_values = None
        
        for step in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                if step > 0 and past_key_values is not None:
                    model_input = input_ids[:, -1:]
                else:
                    model_input = input_ids
                
                output = self.model(model_input, use_cache=True, past_key_values=past_key_values)
                logits = output["logits"][:, -1, :]
                past_key_values = output.get("past_key_values")
            
            # Apply length penalty
            logits = logits / (step + 1) ** length_penalty
            
            # Apply repetition penalty
            for i in range(batch_size * num_beams):
                for token_id in set(input_ids[i].cpu().tolist()):
                    logits[i, token_id] /= repetition_penalty
            
            # Get log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            vocab_size = log_probs.shape[-1]
            
            # Reshape scores for beam search
            # (batch_size * num_beams, vocab_size) -> (batch_size, num_beams * vocab_size)
            log_probs = log_probs.view(batch_size, num_beams * vocab_size)
            
            # Add beam scores
            next_log_probs = log_probs + beam_scores.view(batch_size, num_beams).unsqueeze(-1)
            
            # Select top-k candidates
            next_log_probs = next_log_probs.view(batch_size, -1)
            topk_log_probs, topk_indices = torch.topk(next_log_probs, num_beams, dim=-1)
            
            # Track next token IDs and beam indices
            next_beam_indices = topk_indices // vocab_size
            next_token_ids = topk_indices % vocab_size
            
            # Update input_ids and beam_scores
            next_input_ids = []
            for batch_idx in range(batch_size):
                batch_input = input_ids[batch_idx * num_beams : (batch_idx + 1) * num_beams]
                batch_next = batch_input[next_beam_indices[batch_idx]]
                batch_next = torch.cat([batch_next, next_token_ids[batch_idx].unsqueeze(-1)], dim=-1)
                next_input_ids.append(batch_next)
            
            input_ids = torch.cat(next_input_ids, dim=0)
            beam_scores = topk_log_probs.view(-1)
        
        # Select best beam
        best_indices = beam_scores.view(batch_size, num_beams).argmax(dim=-1)
        best_input_ids = []
        for batch_idx, beam_idx in enumerate(best_indices):
            best_input_ids.append(input_ids[batch_idx * num_beams + beam_idx])
        
        return torch.stack(best_input_ids, dim=0)
    
    def _calculate_score(self, generated_ids: torch.Tensor) -> float:
        """Calculate generation score based on log-probabilities"""
        with torch.no_grad():
            output = self.model(generated_ids)
            logits = output["logits"]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs,
                -1,
                generated_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            
            score = token_log_probs.sum(dim=-1).mean().item()
            return score
    
    def generate_streaming(
        self,
        prompt: str,
        language: str = "en",
        max_new_tokens: int = 100,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        """
        Generate text with streaming output (yields tokens incrementally)
        Berguna untuk real-time output
        """
        with torch.no_grad():
            input_ids = self._prepare_input(prompt, language)
            
            past_key_values = None
            generated_ids = input_ids.clone()
            
            for step in range(max_new_tokens):
                # Forward pass
                if step > 0 and past_key_values is not None:
                    model_input = generated_ids[:, -1:]
                else:
                    model_input = generated_ids
                
                output = self.model(
                    model_input,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                
                logits = output["logits"][:, -1, :]
                past_key_values = output.get("past_key_values")
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-P sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[:, 0] = False
                
                logits_mask = logits.clone()
                logits_mask[:, sorted_indices[sorted_indices_to_remove]] = float('-inf')
                
                probs = torch.softmax(logits_mask, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Yield token
                token = self.tokenizer.decode([next_token.item()])
                yield token


class BatchGenerator:
    """Generate text untuk multiple prompts secara batch"""
    
    def __init__(self, text_generator: TextGenerator):
        self.text_generator = text_generator
    
    def generate_batch(
        self,
        prompts: List[str],
        language: str = "en",
        **kwargs
    ) -> List[Tuple[str, Optional[float]]]:
        """
        Generate completions untuk multiple prompts
        Args:
            prompts: List of prompt strings
            language: Language code
            **kwargs: Additional arguments to pass to generate()
        
        Returns:
            List of (generated_text, score) tuples
        """
        results = []
        for prompt in prompts:
            result = self.text_generator.generate(
                prompt,
                language=language,
                **kwargs
            )
            results.append(result)
        return results
