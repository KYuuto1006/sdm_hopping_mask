#This code is modified from https://github.com/bloc97/CrossAttentionControl
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from clip_modeling import CLIPModel
from diffusers import AutoencoderKL, UNet2DConditionModel
import random
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
import os

auth_token = "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" #input your hugging face access token

#Init CLIP tokenizer and model
model_path_clip = "openai/clip-vit-large-patch14" 
clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)

clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
clip = clip_model.text_model

#Init diffusion model
model_path_diffusion = "CompVis/stable-diffusion-v1-4" #we use v1-4 in the paper
unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)

#Move to GPU
device = "cuda"
unet.to(device)
vae.to(device)
clip.to(device)
print("Loaded all models")

def cal_len(tensor):
    for index, value in enumerate(tensor):
        if value == 49407:
            return index-1 

def use_last_tokens_attention_weights(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_weights = use




@torch.no_grad()
def hopping(prompt="", guidance_scale=7.5, steps=50, seed=None, width=512, height=512):
    #Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 64
    height = height - height % 64
    
    #If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)
    
    #Set inference timesteps to scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps)

    init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
    t_start = 0
    
    #Generate random normal noise
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    #latent = scheduler.add_noise(init_latent, noise, t_start).to(device)
    
    #Process clip
    with autocast(device):
        
        tokens_unconditional = clip_tokenizer("", padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

        full_prompt = prompt
        prompt_split = prompt.split("<|endoftext|>")
        prompt1, prompt2, prompt3, prompt4 = prompt_split[0], prompt_split[1], prompt_split[2], prompt_split[3]

        tokens_prompt1 = clip_tokenizer(prompt1, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        len_prompt1 = cal_len(tokens_prompt1.input_ids[0])
        embedding_prompt1 = clip(tokens_prompt1.input_ids.to(device)).last_hidden_state

        tokens_prompt2 = clip_tokenizer(prompt2, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        len_prompt2 = cal_len(tokens_prompt2.input_ids[0])
        embedding_prompt2 = clip(tokens_prompt2.input_ids.to(device)).last_hidden_state

        tokens_prompt3 = clip_tokenizer(prompt3, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        len_prompt3 = cal_len(tokens_prompt3.input_ids[0])
        embedding_prompt3 = clip(tokens_prompt3.input_ids.to(device)).last_hidden_state

        tokens_prompt4 = clip_tokenizer(prompt4, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        len_prompt4 = cal_len(tokens_prompt4.input_ids[0])
        embedding_prompt4 = clip(tokens_prompt4.input_ids.to(device)).last_hidden_state


        tokens_full_prompt = clip_tokenizer(full_prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_full_prompt = clip(tokens_full_prompt.input_ids.to(device)).last_hidden_state 
        
        embedding_prompt1[0, 1:len_prompt1+1] = embedding_full_prompt[0, 1:len_prompt1+1]
        embedding_prompt2[0, 1:len_prompt2+1] = embedding_full_prompt[0, len_prompt1+2:len_prompt1+len_prompt2+2]
        embedding_prompt3[0, 1:len_prompt3+1] = embedding_full_prompt[0, len_prompt1+len_prompt2+3:len_prompt1+len_prompt2+len_prompt3+3]
        embedding_prompt4[0, 1:len_prompt4+1] = embedding_full_prompt[0, len_prompt1+len_prompt2+len_prompt3+4:len_prompt1+len_prompt2+len_prompt3+len_prompt4+4]
            
        timesteps = scheduler.timesteps[t_start:]
        for number in range(1,5):
            latent = scheduler.add_noise(init_latent, noise, t_start).to(device)
            if number == 1:
                embedding_final = embedding_prompt1
                name = str(number) + prompt1
            elif number == 2:
                embedding_final = embedding_prompt2
                name = str(number) + prompt2
            elif number == 3:
                embedding_final = embedding_prompt3
                name = str(number) + prompt3
            else:
                embedding_final = embedding_prompt4   
                name = str(number) + prompt4        

            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                t_index = t_start + i

                sigma = scheduler.sigmas[t_index]
                latent_model_input = latent
                latent_model_input = (latent_model_input / ((sigma**2 + 1) ** 0.5)).to(unet.dtype)

                #Predict the unconditional noise residual
                noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
            

                use_last_tokens_attention_weights()
                
                #Predict the conditional noise residual and save the cross-attention layer activations
                noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_final).sample
            
                
                #Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latent = scheduler.step(noise_pred, t_index, latent).prev_sample

            #scale and decode the image latents with vae
            latent = latent / 0.18215
            image = vae.decode(latent.to(vae.dtype)).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image[0] * 255).round().astype("uint8")
            Image.fromarray(image).save(os.path.join(r'results/', '{seed} {name}.png'.format(seed=seed, name=name)))



######################################################################

def prompt_token(prompt, index):
    tokens = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True).input_ids[0]
    return clip_tokenizer.decode(tokens[index:index+1])

def show_token_indices(prompt):
    tokens = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True).input_ids[0]
    
    for index in range(len(tokens)):
        decoded_token = clip_tokenizer.decode(tokens[index:index+1])
        print(f'{index}:', decoded_token)
        if decoded_token == "<|endoftext|>":
            break


prompt = "John is working in his farm. <|endoftext|> John is spreading seeds in his farm. <|endoftext|> John is watering crops. <|endoftext|> ...."
# <|endoftext|> is served as separater. Compare the result with single prompt. eg. John is spreading seeds in his farm.
# Our code can support multiple prompts by adding <|endoftext|>, as long as the length of tokens is less than 77. (the restriction of clip encoder)

seed = 1212
#stablediffusion(prompt1, seed=seed, steps=50).save(os.path.join(r'results/mh_results/', 'all zero {seed} {name}.png'.format(seed=seed, name=prompt1)))
#stablediffusion(prompt3, seed=seed, steps=50).save(os.path.join(r'results/mh_results/', 'all zero {seed} {name}.png'.format(seed=seed, name=prompt3)))
#stablediffusion2(prompt=prompt2, prompt2=prompt3, seed=seed, steps=50).save(os.path.join(r'results/mh_results/', 'mh {seed} {name}.png'.format(seed=seed, name=prompt3)))
hopping(prompt, seed=seed, steps=50)
#stablediffusion(prompt1, prompt2, prompt_edit_spatial_start=prompt_edit_spatial_start, prompt_edit_token_weights=prompt_edit_token_weights, seed=seed).save(os.path.join(r'results/ptp_results/', 'new {seed} {start} {weight} {name}.png'.format(seed=seed, start=prompt_edit_spatial_start, weight=prompt_edit_token_weights, name=prompt2)))
#prompt_edit_spatial_start 越大， 越不按照原图