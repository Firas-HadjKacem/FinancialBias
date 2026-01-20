import torch
import numpy as np
from typing import Dict, Any
import math
import transformers
from BiasEvaluation.core.base import ModelBase
import traceback
from transformers import (
      BitsAndBytesConfig,
      AutoModelForCausalLM,
      LlamaTokenizer,
      AutoTokenizer,
      LlamaForCausalLM
  )
from torch.nn.functional import log_softmax
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

class BERTModel(ModelBase):
    """Model wrapper for BERT-based classifiers"""
   
    def __init__(self, model, tokenizer, id2label=None, max_length=512):
        """
        Initialize BERT-based classifier       
        Args:
            model: BERT-based financial classifier model: FinBert, DeBERTa, DistilRoBERTa, etc.,
            tokenizer: BERT tokenizer
            id2label: Label mapping dictionary
            max_length: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = model.device
       
        if torch.cuda.is_available():
            if not str(self.device).startswith('cuda'):
                print(f"Warning: Model not on GPU. Moving to GPU...")
                self.model = self.model.cuda()
                self.device = self.model.device
            print(f"Model running on: {self.device}")
       
        # Set label mapping
        self.id2label = id2label or getattr(model.config, "id2label", {0: "positive", 1: "negative", 2: "neutral"})
   
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate prediction for prompt with probabilities
       
        Args:
            prompt: Input text
           
        Returns:
            Dictionary containing predicted label and probabilities
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        # Move to model's device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
       
        pred_idx = torch.argmax(logits, dim=1).item()  
        # Get label string
        if pred_idx in self.id2label:
            predicted_label = self.id2label[pred_idx]
        elif str(pred_idx) in self.id2label:
            predicted_label = self.id2label[str(pred_idx)]
        else:
            predicted_label = str(pred_idx)
       
        result = {
            "label": predicted_label,
            "probabilities": {self.id2label[i] if i in self.id2label else (self.id2label[str(i)] if str(i) in self.id2label else str(i)):
                             float(prob) for i, prob in enumerate(probabilities)}
        }        
        return result

    def generate_batch(self, prompts):
        """Generate predictions for multiple prompts at once"""
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        pred_idxs = np.argmax(probs, axis=1)
        results = []
        for i in range(len(prompts)):
            pred_idx = pred_idxs[i]
            if pred_idx in self.id2label:
                predicted_label = self.id2label[pred_idx]
            elif str(pred_idx) in self.id2label:
                predicted_label = self.id2label[str(pred_idx)]
            else:
                predicted_label = str(pred_idx)
            results.append({
                "label": predicted_label,
                "probabilities": {self.id2label[j] if j in self.id2label else (self.id2label[str(j)] if str(j) in self.id2label else str(j)): float(probs[i][j]) for j in range(len(probs[i]))}
            })
        return results

class LlamaModelWrapper:
    """
    Wrapper for quantized Llama financial models that predict sentiment using fixed label tokens.
    """
    def __init__(self, model, tokenizer, label_ids, max_length=512):
        """
        label_ids: dict mapping label names (e.g., 'positive') to tokenizer IDs
        """
        self.model = model
        self.tokenizer = tokenizer
        self.label_ids = label_ids  # e.g., {'positive': 6374, ...}
        self.max_length = max_length
        self.device = model.device
        vocab_size = self.tokenizer.vocab_size
        if (self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id < 0 or self.tokenizer.pad_token_id >= vocab_size):
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(2)
            self.tokenizer.pad_token_id = 2

    # Debug helper
    def _print_topk_for_step(self, step_logits, tokenizer, k=30, header=None):
        if header:
            print(header)
        topk_vals, topk_idx = torch.topk(step_logits, k=min(k, step_logits.shape[-1]))
        print("\n[DEBUG] Top tokens at this step:")
        for rank in range(topk_vals.numel()):
            tid = topk_idx[rank].item()
            tok = tokenizer.decode([tid])
            print(f"{rank+1:2d}. id {tid:>5}: {repr(tok)} (logit={topk_vals[rank].item():.4f})")

    # Build label token sequences dynamically
    def _build_label_sequences(self, tokenizer):
        variants = {
            "Positive": [" positive", "positive", "Positive", " positive.", "Positive."],
            "Negative": [" negative", "negative", "Negative", " negative.", "Negative."],
            "Neutral":  [" neutral",  "neutral",  "Neutral",  " neutral.",  "Neutral."],
        }
        seqs = {}
        for lab, forms in variants.items():
            seen, cand = set(), []
            for s in forms + [lab.lower()]:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    t = tuple(ids)
                    if t not in seen:
                        seen.add(t)
                        cand.append(ids)
            seqs[lab] = cand
        return seqs

    # Span finder over generated token ids
    def _find_label_span(self, new_ids, label_seqs):
        best = (None, None, None)  # (label, start_pos, seq_used)
        n = len(new_ids)
        for label, seq_list in label_seqs.items():
            for seq in seq_list:
                m = len(seq)
                if m == 0 or m > n:
                    continue
                for i in range(0, n - m + 1):
                    if new_ids[i:i+m] == seq:
                        if best[1] is None or i < best[1]:
                            best = (label, i, seq)
                        break
        return best

    # build label-id sets from label mapping
    def _build_label_id_sets(self):
        # {"Positive":[6374], "Negative":[8178,22198], "Neutral":[21104]}
        lab_sets = {"Positive": set(), "Negative": set(), "Neutral": set()}
        for k, ids in self.label_ids.items():
            lab = k.capitalize()
            for t in (ids if isinstance(ids, list) else [ids]):
                lab_sets[lab].add(int(t))
        union = set().union(*lab_sets.values())
        return lab_sets, union

    # Logits processor to force label on the FIRST step
    class FirstStepLabelOnly(LogitsProcessor):
        """
        At the FIRST generation step, allow only tokens that are valid FIRST tokens
        of any label variant (e.g., 'positive', 'negative', 'neutral', or cased/dotted forms).
        Later steps are unconstrained.
        """
        def __init__(self, allowed_first_token_ids):
            super().__init__()
            self.allowed = None
            if allowed_first_token_ids:
                self.allowed = torch.tensor(sorted(set(allowed_first_token_ids)), dtype=torch.long)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if self.allowed is None:
                return scores
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.allowed] = 0.0
            return scores + mask

    def _restricted_label_softmax(self, step_logits):
        """
        Compute P(label | step) using only the label token logits.
        Handles multi-id Negative via log-sum-exp over its ids (since Negative can be decoded from multiple ids in our tokenizer).
        """
        pos_ids = self.label_ids["Positive"] if isinstance(self.label_ids["Positive"], list) else [self.label_ids["Positive"]]
        neg_ids = self.label_ids["Negative"] if isinstance(self.label_ids["Negative"], list) else [self.label_ids["Negative"]]
        neu_ids = self.label_ids["Neutral"]  if isinstance(self.label_ids["Neutral"],  list) else [self.label_ids["Neutral"]]

        # pull logits
        v_pos = step_logits[pos_ids[0]].item() 
        v_neu = step_logits[neu_ids[0]].item() 

        # Negative can have multiple ids -> log-sum-exp across them
        neg_vec = step_logits[torch.tensor(neg_ids, dtype=torch.long, device=step_logits.device)]
        v_neg = torch.logsumexp(neg_vec, dim=0).item()

        # softmax across the three label scores
        m = max(v_pos, v_neg, v_neu)
        s_pos = math.exp(v_pos - m)
        s_neg = math.exp(v_neg - m)
        s_neu = math.exp(v_neu - m)
        Z = s_pos + s_neg + s_neu

        probs = {
            "Positive": s_pos / Z,
            "Negative": s_neg / Z,
            "Neutral":  s_neu / Z,
        }
        return probs


    def generate(self, prompt, debug=True, topk=30, enforce_label_first_token=True):
        tokenizer, model, device = self.tokenizer, self.model, self.device

        # Build label text variants and allowed first-token ids (for step-0 constraint)
        label_seqs = self._build_label_sequences(tokenizer)
        allowed_first_ids = list({seq[0] for seqs in label_seqs.values() for seq in seqs if len(seq) > 0})

        # Label id sets and skip-set (EOS + empty)
        label_id_sets, label_union = self._build_label_id_sets()
        EOS_TID = getattr(tokenizer, "eos_token_id", 2)
        EMPTY_TID = 29871
        SKIP_TIDS = {EOS_TID, EMPTY_TID}

        if debug:
            print(f"Processing 1 prompt")

        try:
            enc = tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,          
                truncation=True,
                max_length=self.max_length
            ).to(device)

            lp = None
            if enforce_label_first_token:
                lp = LogitsProcessorList([self.FirstStepLabelOnly(allowed_first_ids)])

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=2,        
                    min_new_tokens=1,        
                    do_sample=False,        
                    output_scores=True,
                    return_dict_in_generate=True,
                    logits_processor=lp,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                    pad_token_id=getattr(tokenizer, "eos_token_id", None),
                )

            sequences   = out.sequences # [1, seq_len]
            scores_list = out.scores # list len==gen_steps; each [1, V]
            gen_steps   = len(scores_list)

            seq_ids_all = sequences[0].tolist()
            gen_ids     = seq_ids_all[-gen_steps:] if gen_steps > 0 else []

            answer_part = tokenizer.decode(gen_ids,    skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            full_text   = tokenizer.decode(seq_ids_all, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            if debug:
                print(f"\n— Prompt [0] generated answer: {repr(answer_part)}  gen_ids={gen_ids}")

            # pick the first sentiment token id within the generated window, skipping EOS/empty
            pos = None
            for i, tid in enumerate(gen_ids):
                tid = int(tid)
                if tid in SKIP_TIDS:
                    continue
                if tid in label_union:
                    pos = i
                    if debug:
                        print(f"[ANCHOR] pos={pos} (tid={tid}) within generated window; skipped {SKIP_TIDS}")
                    break

            # if still not found, try text span finder among variants (within the generated window)
            if pos is None and gen_steps > 0:
                label_found_span, pos_span, _ = self._find_label_span(gen_ids, label_seqs)
                if (label_found_span is not None) and (pos_span is not None) and (pos_span < gen_steps):
                    pos = pos_span
                    if debug:
                        print(f"[ANCHOR] pos={pos} (from span finder in generated window)")

            # Scoring at anchor step or fallback
            if pos is not None and gen_steps > 0 and pos < gen_steps:
                step_logits = scores_list[pos][0]
                prob_dict = self._restricted_label_softmax(step_logits)
                logits_sentiment = max(prob_dict, key=prob_dict.get)

                if debug:
                    self._print_topk_for_step(step_logits, tokenizer, k=topk,
                        header=f"\n==== TOP-K (ANCHOR STEP {pos}) ====")
                    print(f"[P(Positive), P(Negative), P(Neutral)] = "
                        f"{prob_dict['Positive']}, {prob_dict['Negative']}, {prob_dict['Neutral']}")

            else:
                # fallback: use first step’s logits
                if gen_steps == 0:
                    prob_dict = {"Positive": 1/3, "Negative": 1/3, "Neutral": 1/3}
                    logits_sentiment = "Neutral"
                else:
                    step0 = scores_list[0][0]
                    if debug:
                        self._print_topk_for_step(step0, tokenizer, k=topk,
                            header="\n==== FIRST-STEP FALLBACK TOP-K ====")
                    prob_dict = self._restricted_label_softmax(step0)
                    logits_sentiment = max(prob_dict, key=prob_dict.get)
                pos = 0

            # surface label from generated text
            al = answer_part.lower()
            if   "positive" in al: text_label = "Positive"
            elif "negative" in al: text_label = "Negative"
            elif "neutral"  in al: text_label = "Neutral"
            else:                   text_label = "NA"

            is_match = (text_label == logits_sentiment)
            if debug:
                print(f"\n[RESULT] text={text_label}  logits={logits_sentiment}  match={is_match}")

            return {
                "label": text_label,
                "probabilities": prob_dict,
                "generated_text": full_text,
                "answer_part": answer_part,
                "sentiment_position": pos,
                "match": is_match,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "label": "ERROR",
                "probabilities": {"Positive": 1/3, "Negative": 1/3, "Neutral": 1/3},
                "generated_text": f"Error: {str(e)}",
                "answer_part": "",
                "sentiment_position": 0,
                "match": False,
            }

    def generate_batch(self, prompts, batch_size=128, debug=True, topk=30, enforce_label_first_token=True):
        tokenizer, model, device = self.tokenizer, self.model, self.device
        label_seqs = self._build_label_sequences(tokenizer)

        # Allowed first-token ids: first id of every variant of every label
        allowed_first_ids = list({seq[0] for seqs in label_seqs.values() for seq in seqs if len(seq) > 0})

        # Label id sets and skip-set
        label_id_sets, label_union = self._build_label_id_sets()
        EOS_TID = getattr(tokenizer, "eos_token_id", 2)
        EMPTY_TID = 29871
        SKIP_TIDS = {EOS_TID, EMPTY_TID}

        if debug:
            print(f"Processing {len(prompts)} prompts with batch_size={batch_size}")

        all_results = []
        true_matches = 0
        false_matches = 0
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start+batch_size]
            if debug:
                print(f"\nProcessing batch {start//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} "
                    f"({len(batch_prompts)} prompts)")

            try:
                batch_inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,          
                    truncation=True,
                    max_length=self.max_length
                ).to(device)

                input_lengths = batch_inputs["attention_mask"].sum(dim=1).tolist()

                lp = None
                if enforce_label_first_token:
                    lp = LogitsProcessorList([self.FirstStepLabelOnly(allowed_first_ids)])

                with torch.no_grad():
                    outputs = model.generate(
                        **batch_inputs,
                        max_new_tokens=2,       
                        min_new_tokens=1,       
                        do_sample=False,
                        output_scores=True,
                        return_dict_in_generate=True,
                        logits_processor=lp,
                        eos_token_id=getattr(tokenizer, "eos_token_id", None),
                        pad_token_id=getattr(tokenizer, "eos_token_id", None)
                    )

                sequences    = outputs.sequences                    # [B, in_len + gen_len]
                scores_list  = outputs.scores                       # list len==gen_len; each [B, V]
                gen_steps    = len(scores_list)
                logprob_list = [log_softmax(s, dim=-1) for s in scores_list] if gen_steps > 0 else []

                bsz_now = sequences.size(0)
                assert bsz_now == len(batch_prompts)

                for b in range(bsz_now):
                    seq_ids_all = sequences[b].tolist()

                    gen_ids = seq_ids_all[-gen_steps:] if gen_steps > 0 else []

                    answer_part = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                    full_text   = tokenizer.decode(seq_ids_all, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                    if debug:
                        print(f"\n— Prompt [{b}] generated answer: {repr(answer_part)}  gen_ids={gen_ids}")

                    # pick the first label token within the generated window, skipping {eos, ''}
                    pos = None
                    for i, tid in enumerate(gen_ids):
                        tid = int(tid)
                        if tid in SKIP_TIDS:
                            continue
                        if tid in label_union:
                            pos = i
                            if debug: print(f"[ANCHOR] pos={pos} (tid={tid}) within generated window; skipped {SKIP_TIDS}")
                            break

                    # If still not found, try span finder inside the generated window
                    if pos is None and gen_steps > 0:
                        label_found_span, pos_span, _ = self._find_label_span(gen_ids, label_seqs)
                        if (label_found_span is not None) and (pos_span is not None) and (pos_span < gen_steps):
                            pos = pos_span
                            if debug: print(f"[ANCHOR] pos={pos} (from span finder in generated window)")

                    if pos is not None and gen_steps > 0 and pos < gen_steps:
                        step_logits = scores_list[pos][b]
                        prob_dict = self._restricted_label_softmax(step_logits)
                        logits_sentiment = max(prob_dict, key=prob_dict.get)

                        if debug:
                            self._print_topk_for_step(step_logits, tokenizer, k=topk,
                                header=f"\n==== TOP-K (ANCHOR STEP {pos}) ====")
                            print(f"[P(Positive), P(Negative), P(Neutral)] = "
                                f"{prob_dict['Positive']}, {prob_dict['Negative']}, {prob_dict['Neutral']}")
 
                        # surface label from text
                        al = answer_part.lower()
                        if   "positive" in al: text_label = "Positive"
                        elif "negative" in al: text_label = "Negative"
                        elif "neutral"  in al: text_label = "Neutral"
                        else:                   text_label = "NA"

                        is_match = (text_label == logits_sentiment)  # NEW

                        if debug:
                            print(f"\n[RESULT] text={text_label}  logits={logits_sentiment}  match={text_label==logits_sentiment}")

                        if is_match: true_matches += 1
                        else:        false_matches += 1

                        all_results.append({
                            "label": text_label,
                            "probabilities": prob_dict,
                            "generated_text": full_text,
                            "answer_part": answer_part,
                            "sentiment_position": pos if pos is not None else 0,
                            "match": (text_label == logits_sentiment),
                        })

                    else:
                        # fallback using first step
                        if gen_steps == 0:
                            prob_dict = {"Positive": 1/3, "Negative": 1/3, "Neutral": 1/3}
                            logits_sentiment = "NG"
                        else:
                            step0 = scores_list[0][b]
                            if debug:
                                self._print_topk_for_step(step0, tokenizer, k=topk,
                                    header="\n==== FIRST-STEP FALLBACK TOP-K ====")
                            prob_dict = self._restricted_label_softmax(step0)
                            logits_sentiment = max(prob_dict, key=prob_dict.get)
                        al = answer_part.lower()
                        if   "positive" in al: text_label = "Positive"
                        elif "negative" in al: text_label = "Negative"
                        elif "neutral"  in al: text_label = "Neutral"
                        else:                   text_label = "NA"
                        is_match = (text_label == logits_sentiment)  

                        if debug:
                            print(f"\n[RESULT] (fallback) text={text_label}  logits={logits_sentiment}  match={text_label==logits_sentiment}")
                        if is_match: true_matches += 1
                        else:        false_matches += 1
                        all_results.append({
                            "label": text_label,
                            "probabilities": prob_dict,
                            "generated_text": full_text,
                            "answer_part": answer_part,
                            "sentiment_position": 0,
                            "match": (text_label == logits_sentiment),
                        })

            except Exception as e:
                traceback.print_exc()
                all_results.extend([
                    {
                        "label": "ERROR",
                        "probabilities": {"Positive": 1/3, "Negative": 1/3, "Neutral": 1/3},
                        "generated_text": f"Error in batch {start//batch_size + 1}: {str(e)}",
                        "answer_part": ""
                    }
                    for _ in batch_prompts
                ])

        if debug:
            total = true_matches + false_matches
            acc = (true_matches / total) if total else 0.0
            print(f"\n[STATS] match=True: {true_matches} | match=False: {false_matches} |" 
                f"accuracy={acc:.3%} over {total} scored items") 
        return all_results


def load_llama_model(base_tokenizer_id, model_id, device_map="auto", **kwargs):
    """
    Loads a quantized Llama model with tokenizer, bypassing auto-detection.
    """    
    # Load the tokenizer
    try:
        tok = LlamaTokenizer.from_pretrained(base_tokenizer_id, **kwargs)
    except Exception as e:
        print(f"LlamaTokenizer failed: {e}, trying AutoTokenizer...")
        tok = AutoTokenizer.from_pretrained(base_tokenizer_id, **kwargs)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
    # Load the model with explicit class instead of Auto
    try:
        # Try loading with BitsAndBytesConfig
        try:            
            mod = LlamaForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_safetensors=True,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                device_map=device_map,
                **kwargs
            )
            
        except (ImportError, AttributeError):
            # Direct params approach
            mod = LlamaForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_safetensors=True,
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map=device_map,
                **kwargs
            )
            
    except Exception as e:
        print(f"Failed to load with LlamaForCausalLM: {e}")
        # As a last resort, use AutoModel with config_overrides        
        mod = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **kwargs
        )
        
    print(f"Model loaded successfully to {device_map}")
    return mod, tok
  

def load_bert_model(model_name: str):
    """
    Load bert-based model and tokenizer
   
    Args:
        model_name: HuggingFace model name
       
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, tokenizer