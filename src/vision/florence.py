import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from typing import Optional, Dict, Any

from src.vision.vision_engine import VisionEngine
from src.config import CONFIG
from src.vision.utils import preprocess_image, auto_enhance
from src.vision.captioning import format_caption
from src.vision.ocr import format_ocr
from src.vision.detection import format_object_detection

try:
    import spaces
    IS_SPACES = True
    gpu_decorator = spaces.GPU
except ImportError:
    IS_SPACES = False
    def gpu_decorator(func):
        return func

def get_device() -> str:
    """Select best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE: str = "cpu" if IS_SPACES else get_device()
DTYPE: torch.dtype = torch.float16 if IS_SPACES or DEVICE == "cuda" else torch.float32

class FlorenceVisionEngine(VisionEngine):
    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.processor: Optional[AutoProcessor] = None
        self.paddle_ocr = None

    def load(self):
        """Load the Florence-2 model."""
        if self.model is not None:
            return
            
        try:
            print(f"Loading Florence-2 on {DEVICE.upper()} (will move to GPU during inference if on Spaces)...")
            
            # Hotfix for Florence-2 in newer transformers versions
            import transformers
            if not hasattr(transformers.PretrainedConfig, "forced_bos_token_id"):
                transformers.PretrainedConfig.forced_bos_token_id = None
            
            # Force _supports_sdpa to False on the actual base class
            import transformers.modeling_utils
            transformers.modeling_utils.PreTrainedModel._supports_sdpa = False
                
            self.model = AutoModelForCausalLM.from_pretrained(
                CONFIG.MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=DTYPE,
                attn_implementation="eager"
            ).eval()

            # Hotfix for Florence-2 processor tokenizer compatibility
            if not hasattr(transformers.PreTrainedTokenizerBase, "additional_special_tokens"):
                transformers.PreTrainedTokenizerBase.additional_special_tokens = property(
                    lambda self: getattr(self, "_additional_special_tokens", [])
                )

            self.processor = AutoProcessor.from_pretrained(
                CONFIG.MODEL_NAME,
                trust_remote_code=True,
            )
            print("Florence-2 loaded successfully")
            
            try:
                from paddleocr import PaddleOCR
                print("Loading PaddleOCR...")
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                print("PaddleOCR loaded successfully")
            except Exception as e:
                print(f"PaddleOCR load failed: {e}")
                
            self._warmup()
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def _warmup(self):
        """Run a dummy inference to warm up kernels."""
        if IS_SPACES:
            print("Skipping warmup on ZeroGPU Spaces")
            return
        try:
            dummy = Image.new("RGB", (224, 224), 128)
            self._run_inference(dummy, "<CAPTION>")
            print("Model warmed up")
        except Exception as e:
            print(f"Warmup warning: {e}")

    @gpu_decorator
    def _run_inference(self, image: Image.Image, task_token: str, text_input: str = None) -> Dict[str, Any]:
        """Core inference logic."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Ensure model is on the right device when inference runs
        target_device = "cuda" if IS_SPACES else DEVICE
        if next(self.model.parameters()).device.type != target_device:
            self.model.to(target_device)

        image = preprocess_image(image)
        image = auto_enhance(image)
        max_tokens = CONFIG.MAX_NEW_TOKENS.get(task_token, 64)

        prompt = task_token if text_input is None else task_token + text_input

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(target_device)
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

        raw_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        result = self.processor.post_process_generation(
            raw_text,
            task=task_token,
            image_size=(image.width, image.height),
        )
        return result

    def ask_question(self, image: Image.Image, question: str) -> str:
        try:
            result = self._run_inference(image, "<VQA>", text_input=question)
            answer = result.get("<VQA>", "")
            if not answer:
                return "I couldn't find an answer to that."
            
            # Florence-2 sometimes outputs spatial coordinates in VQA (e.g. <loc_123>, <poly>)
            # We must strip these out for text-to-speech
            import re
            clean_answer = re.sub(r'<[^>]+>', '', answer).strip()
            
            if not clean_answer:
                return "I couldn't see that clearly."
                
            return clean_answer
        except Exception as e:
            print(f"ask_question error: {e}")
            return "I couldn't analyze the question right now."

    def describe_scene(self, image: Image.Image, detailed: bool = False) -> str:
        task = "<MORE_DETAILED_CAPTION>" if detailed else "<DETAILED_CAPTION>"
        try:
            result = self._run_inference(image, task)
            return format_caption(result.get(task, ""))
        except Exception as e:
            print(f"describe_scene error: {e}")
            return "I couldn't analyze the scene right now."

    def read_text(self, image: Image.Image) -> str:
        try:
            if hasattr(self, 'paddle_ocr') and self.paddle_ocr:
                import numpy as np
                # Convert PIL Image to RGB Numpy array for PaddleOCR
                img_array = np.array(image.convert("RGB"))
                result = self.paddle_ocr.ocr(img_array, cls=True)
                
                if not result or result[0] is None:
                    return "I couldn't find any clear text in the image."
                
                lines = []
                for line in result[0]:
                    text = line[1][0]
                    lines.append(text)
                
                final_text = " ".join(lines).strip()
                if not final_text:
                    return "I couldn't find any clear text."
                return f"The text says: {final_text}"
            else:
                # Fallback to Florence-2 OCR
                result = self._run_inference(image, "<OCR>")
                return format_ocr(result.get("<OCR>", ""))
        except Exception as e:
            print(f"read_text error: {e}")
            return "I couldn't read the text right now."

    def analyze(self, image: Image.Image, task: str) -> str:
        # Generic handler
        try:
            result = self._run_inference(image, task)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
