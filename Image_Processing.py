from typing import List
from torchvision.ops import box_convert
import subprocess
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from diffusers import StableDiffusionInpaintPipeline
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import matplotlib.pyplot as plt
import os
import supervision as sv
from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import cv2

#SEQUENCE OF COMMANDS TO BE EXECUTED IN. You have to check alternative of cd in windows i.e chdir
# 1: '%cd{HOME_DIR}',
# 2: '!git clone https: // github.com / IDEA - Research / GroundingDINO.git'
# 3: '%cd {HOME_DIR} / GroundingDINO',
# 4:'!pip install - q - e.', '!pip install - q roboflow'
# 5: '%cd {HOME}'
# 6:'!mkdir {HOME_DIR} / weights',
# 7: '%cd {HOME_DIR} / weights',
# 8: '!wget - q https: // github.com / IDEA - Research / GroundingDINO / releases / download / v0.1.0 - alpha / groundingdino_swint_ogc.pth'
# 9: '!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
# 10: '!{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git' you need to run this command to intsall segment anything model
# 11: '!pip uninstall -y supervision'
# 12: '!pip install -q -U supervision==0.6.0'


# Get the current working directory
HOME_DIR=''#os.getcwd() #need to run '!git clone https: // github.com / IDEA - Research / GroundingDINO.git' command to clone GroundingDINO repo in home dir
GD_DIR=''#os.path.join(HOME_DIR, "GroundingDINO") # need to run '!pip install - q - e.', '!pip install - q roboflow' after moving into GroundingDINO
weights_dir=''#os.path.join(HOME_DIR, "weights") #need to download weights here via command '!wget - q https: // github.com / IDEA - Research / GroundingDINO / releases / download / v0.1.0 - alpha / groundingdino_swint_ogc.pth'
# import sys
# !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git' you need to run this command to intsall segment anything model
# !pip uninstall -y supervision
# !pip install -q -U supervision==0.6.0


'''Using StableDiffusionInpaintPipeline for inpainting task, using
    "stabilityai/stable-diffusion-2-inpainting" model 
    from hugging face, loading model in torch.16 precision.'''

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe.set_use_memory_efficient_attention_xformers(True)
pipe.to("cuda")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)


def outpaint(image, prompt, height, width):
    """
    Perform outpainting on an image using a given prompt.

    Args:
        image (PIL Image): Input image.
        prompt (str): Prompt for outpainting.
        height (int): Desired height for the output image.
        width (int): Desired width for the output image.

    Returns:
        PIL Image: Outpainted image.
    """
    # load the image, extract the mask
    rgba = image  # Image.open('green.jpg')
    # Open the original image
    original_image = rgba  # Image.open("original_image.png")

    # Calculate the dimensions of the new image
    new_width = width  # original_image.width * 2
    new_height = height  # original_image.height * 2

    # Create a new RGBA image with white color and the calculated dimensions
    new_image = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))

    # Calculate the position to paste the original image in the center
    paste_position = (
        (new_width - original_image.width) // 2,
        (new_height - original_image.height) // 2
    )

    # Paste the original image in the center of the new image
    new_image.paste(original_image, paste_position)

    # Convert the alpha channel to truly binary values (0 or 255)
    new_image = new_image.convert("RGBA")
    new_image_data = new_image.getdata()
    new_image_data = [(r, g, b, 0 if a < 128 else 255) for r, g, b, a in new_image_data]
    new_image.putdata(new_image_data)

    # Save the resulting image
    new_image.save("new_image.png", format="PNG")

    rgba = new_image
    # rgba = rgba.convert("RGBA")
    mask_image = Image.fromarray(np.array(rgba)[:, :, 3] == 0)
    # mask_img_pil=Image.fromarray(mask_image)
    mask_image.save("mask.png", "PNG")
    # run the pipeline
    prompt = prompt
    image = pipe(
        prompt=prompt,
        image=rgba,
        mask_image=mask_image,
    ).images[0]
    return image


def generate_prompt(image):
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = image  # Image.open('inp.png').convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    return processor.decode(out[0], skip_special_tokens=True)


def run(image, h, w):
    """
    Run the complete process of generating prompts and outpainting an image.

    Args:
        image (PIL Image): Input image.
        h (int): Desired height for the output image.
        w (int): Desired width for the output image.

    Returns:
        PIL Image: Outpainted image.
    """
    import time
    prompt = generate_prompt(image)
    prompt = prompt + ", 4K, high quality, professional image"
    if h == None:
        h, _ = image.size
    if w == None:
        _, w = image.size
    res = outpaint(image, prompt, int(h), int(w))

    return res,prompt


def crop_including_bbox(image, x_max, y_max, x_min, y_min, h, w):
    x1, y1, x2, y2 = int(x_min), int(y_min), int(x_max), int(y_max)
    if (x_max - x_min) >= 512 and (y_max - y_min) >= 512:
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_image = resize_image(cropped_image, 512, 512)
        return cropped_image
    elif (x_max - x_min) >= 512:
        w = 1
        image = resize_image(image, image.shape[0], 512)
    elif (y_max - y_min) >= 512:
        h = 1
        image = resize_image(image, 512, image.shape[1])
    else:
        image = image
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    # Calculate the center of the detected bounding box
    bbox_center_x = (x_max + x_min) / 2
    bbox_center_y = (y_max + y_min) / 2

    # Calculate the crop box coordinates
    crop_x_min = int(max(bbox_center_x - 256, 0))
    crop_y_min = int(max(bbox_center_y - 256, 0))
    crop_x_max = int(min(bbox_center_x + 256, image.shape[1]))
    crop_y_max = int(min(bbox_center_y + 256, image.shape[0]))

    # Calculate the actual cropping region that includes the bounding box
    actual_crop_x_min = crop_x_min - int(bbox_center_x - x_min)
    actual_crop_y_min = crop_y_min - int(bbox_center_y - y_min)
    actual_crop_x_max = actual_crop_x_min + 512
    actual_crop_y_max = actual_crop_y_min + 512
    if actual_crop_y_min < 0:
        diff = abs(actual_crop_y_min)
        actual_crop_y_min = diff
        actual_crop_y_max = actual_crop_y_max + (2 * diff)
    if actual_crop_x_min < 0:
        diff = abs(actual_crop_x_min)
        actual_crop_x_min = diff
        actual_crop_x_max = actual_crop_x_max + (2 * diff)
    if actual_crop_y_max < y_max:
        actual_crop_y_max = y_max
        actual_crop_y_min = actual_crop_y_min + abs(actual_crop_y_max - actual_crop_y_min - 512)
    if actual_crop_x_max < x_max:
        actual_crop_x_max = x_max
        actual_crop_x_min = actual_crop_x_min + abs(actual_crop_x_max - actual_crop_x_min - 512)
    x_min, y_min, x_max, y_max = actual_crop_x_min, actual_crop_y_min, actual_crop_x_max, actual_crop_y_max
    # Calculate the center of the bounding box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Calculate half the width and height of the 512x512 bounding box
    half_width = 256
    half_height = 256

    # Calculate the top-left and bottom-right coordinates of the 512x512 bounding box
    x_min_512 = max(center_x - half_width, 0)
    y_min_512 = max(center_y - half_height, 0)
    x_max_512 = min(center_x + half_width, image.shape[1])
    y_max_512 = min(center_y + half_height, image.shape[0])

    if x_min_512 < x1:
        x_min_512 = x1
        temp = x_min_512 + 512  # x_max_512+(x1-x_min_512)
        if temp < image.shape[1]:
            x_min_512 = x1
            x_max_512 = temp
        else:
            for i in range(515):
                if (x_min_512 - 1) >= 0:
                    x_min_512 = x_min_512 - 1
                if (x_max_512 + 1) < image.shape[0]:
                    x_max_512 = x_max_512 + 1
                if (x_max_512 - x_min_512) >= 512:
                    break
    if y_min_512 < y1:
        y_min_512 = y1
        temp = y_min_512 + 512  # y_max_512+(y1-y_min_512)
        if temp < image.shape[0]:
            y_min_512 = y1
            y_max_512 = temp
        else:
            for i in range(515):
                if (y_min_512 - 1) >= 0:
                    y_min_512 = y_min_512 - 1
                if (y_max_512 + 1) < image.shape[0]:
                    y_max_512 = y_max_512 + 1
                if (y_max_512 - y_min_512) >= 512:
                    break


    # Calculate the new width and height of the 512x512 bounding box
    new_width = x_max_512 - x_min_512
    new_height = y_max_512 - y_min_512

    # Draw the bounding box on the image
    image_with_box = image.copy()
    cv2.rectangle(image_with_box, (x_min_512, y_min_512),
                  (x_min_512 + new_width, y_min_512 + new_height), (0, 255, 0), 2)

    image_with_box_rgb = image_with_box  # cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_with_box_rgb)
    # plt.axis('off')
    # plt.show()
    if (h == 1 or y_max_512 == 512) and (w == 1 or x_max_512 == 512):
        cropped_image = image[0:512, 0:512]
    elif h == 1 or y_max_512 == 512:
        cropped_image = image[x_min_512:x_min_512 + new_width, 0:512]
    elif w == 1 or x_max_512 == 512:
        cropped_image = image[0:512, y_min_512:y_min_512 + new_height]
    else:
        cropped_image = image[y_min_512:y_min_512 + new_height,
                        x_min_512:x_min_512 + new_width]

    # Calculate the new width and height of the 512x512 bounding box
    # new_width = actual_crop_x_max - actual_crop_x_min
    # new_height = actual_crop_y_max - actual_crop_y_min
    # print(new_width,new_height)

    # print(actual_crop_x_min,actual_crop_y_min,actual_crop_x_max,actual_crop_y_max)

    # cropped_image = image[actual_crop_y_min:actual_crop_y_max,
    #     actual_crop_x_min:actual_crop_x_max]
    # print(cropped_image.shape)

    return cropped_image


def crop_and_draw_bounding_box(image, x_min, y_min, x_max, y_max, h, w):
    if image.shape[0] < 512 and image.shape[1] < 512:
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Calculate the center of the bounding box
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Calculate half the width and height of the 512x512 bounding box
        half_width = 256
        half_height = 256

        # Calculate the top-left and bottom-right coordinates of the 512x512 bounding box
        x_min_512 = max(center_x - half_width, 0)
        y_min_512 = max(center_y - half_height, 0)
        x_max_512 = min(center_x + half_width, image.shape[1])
        y_max_512 = min(center_y + half_height, image.shape[0])

        # Calculate the new width and height of the 512x512 bounding box
        new_width = x_max_512 - x_min_512
        new_height = y_max_512 - y_min_512

        # Draw the bounding box on the image
        image_with_box = image.copy()
        cv2.rectangle(image_with_box, (x_min_512, y_min_512),
                      (x_min_512 + new_width, y_min_512 + new_height), (0, 255, 0), 2)

        image_with_box_rgb = image_with_box  # cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)

        if h == 1 and w == 1:
            cropped_image = image[0:512, 0:512]
        elif h == 1:
            cropped_image = image[y_min_512:y_min_512 + new_height, 0:512]
        elif w == 1:
            cropped_image = image[0:512, x_min_512:x_min_512 + new_width]
        else:
            cropped_image = image[y_min_512:y_min_512 + new_height,
                            x_min_512:x_min_512 + new_width]
    else:
        cropped_image = crop_including_bbox(image, x_max, y_max, x_min, y_min, h, w)

    if cropped_image.shape[0] == 512 and cropped_image.shape[1] == 512:
        if image.shape[0] == 512 and image.shape[1] == 512:
            cropped_image = image
        return cropped_image, ''
    elif cropped_image.shape[0] < 512 or cropped_image.shape[1] < 512:
        cropped_image = Image.fromarray(cropped_image)
        res, prompt = run(cropped_image, 512, 512)
        cropped_image = np.array(res)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("cropped_image.png", cropped_image)
        return cropped_image, prompt
    elif cropped_image.shape[0] > 512 and cropped_image.shape[1] > 512:
        cv2.imwrite("cropped_image.png", cropped_image)
        return cropped_image, ''
    else:
        return cropped_image, ''
    image_with_box_rgb = image_with_box  # cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
    cv2.imwrite("cropped_image.png", cropped_image)


def resize_image(input_path, height=None, width=None, interpolation=cv2.INTER_LINEAR):
    original_image = input_path

    if height is None and width is None:
        # If both height and width are None, return the original image
        return original_image

    # Get the original image dimensions
    orig_height, orig_width, _ = original_image.shape

    if height is None:
        # Calculate the new width while maintaining the aspect ratio
        ratio = width / float(orig_width)
        new_height = int(orig_height * ratio)
        desired_size = (width, new_height)
    elif width is None:
        # Calculate the new height while maintaining the aspect ratio
        ratio = height / float(orig_height)
        new_width = int(orig_width * ratio)
        desired_size = (new_width, height)
    else:
        # Both height and width are specified; use them without preserving aspect ratio
        desired_size = (width, height)

    # Resize the image using OpenCV's resize function with interpolation
    resized_image = cv2.resize(original_image, desired_size, interpolation=interpolation)

    return resized_image


def three_channels_check(image):
    def is_bgr_image(image):
        num_channels = image.shape[2]

        if num_channels == 3 and image[0, 0, 0] < image[0, 0, 2]:

            print("The image is in BGR color space.")
            return True
        else:
            print("The image is not in BGR color space.")
            return False

    def ensure_three_channels(image):
        # Check if the image has 3 channels already
        if image.shape[-1] == 3:
            return image

        # If the image has more than 3 channels, discard the extra channels
        if image.shape[-1] > 3:
            image = image[..., :3]

        # If the image has only 1 channel, replicate it to create three channels
        if image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        return image

    # Check if the image is in BGR format
    is_bgr = is_bgr_image(image)
    print(is_bgr)

    # Convert BGR image to RGB format if necessary
    if is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ensure the image has only three channels (RGB)
    processed_image = ensure_three_channels(image)
    return processed_image


def merge_bounding_boxes(boxes):
    """
        Merge two bounding boxes into a single bounding box that encompasses both.

        Args:
            box1 (list): First bounding box [x_min, y_min, x_max, y_max].
            box2 (list): Second bounding box [x_min, y_min, x_max, y_max].

        Returns:
            list: Merged bounding box [x_min, y_min, x_max, y_max].
        """
    x1, y1, x2, y2 = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
    for i in range(len(boxes)):
        x1 = min(x1, boxes[i][0])
        y1 = min(y1, boxes[i][1])
        x2 = max(x2, boxes[i][2])
        y2 = max(y2, boxes[i][3])
    return [x1, y1, x2, y2]


def run_cmd(command):
    try:
        # Run the CMD command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr

    except subprocess.CalledProcessError as e:
        return False, str(e)



def load_and_download_weights():
    """
        Load and download required weights and configurations.

        Args:
            HOME (str): Path to the home directory.

        Returns:
            None
        """
    # commds = [f'%cd{HOME_DIR}', '!git clone https: // github.com / IDEA - Research / GroundingDINO.git'
    #     , f'%cd {HOME_DIR} / GroundingDINO', '!pip install - q - e.', '!pip install - q roboflow'
    #     , '%cd {HOME}', f'!mkdir {HOME_DIR} / weights', f'%cd {HOME_DIR} / weights',
    #           '!wget - q https: // github.com / IDEA - Research / GroundingDINO / releases / download / v0.1.0 - alpha / groundingdino_swint_ogc.pth'
    #           ]

    for cmd_command in commds:
        success, output = run_cmd(cmd_command)

        if success:
            print(f"Command {cmd_command} output:")
            print(output)
        else:
            print(f"Command {cmd_command} failed with error:")
            print(output)

def enhance_class_name(class_names: List[str]) -> List[str]:
    """
        Enhance class names by adding a prefix.

        Args:
            class_names (List[str]): List of class names.

        Returns:
            List[str]: List of enhanced class names.
        """
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """
        Annotate an image with bounding boxes and labels.

        Args:
            image_source (np.ndarray): Source image.
            boxes (torch.Tensor): Bounding boxes.
            logits (torch.Tensor): Logits.
            phrases (List[str]): List of phrases.

        Returns:
            np.ndarray: Annotated image.
        """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)
    result_array = np.array(detections.xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections,
                                             labels=labels)
    return annotated_frame, result_array


def detect_and_create_bounding_box(image,word_mask,HOME):
    """
        Detect and create a bounding box around an object.

        Args:
            image (PIL Image): Input image.
            word_mask (str): Object to detect.
            HOME (str): Path to the home directory.

        Returns:
            np.ndarray: Array of bounding box coordinates.
        """
    CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
    WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
    WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
    print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
    # cmd_command="%cd {HOME} / GroundingDINO"
    # success, output = run_cmd()
    # if success:
    #     print(f"Command {cmd_command} output:")
    #     print(output)
    # else:
    #     print(f"Command {cmd_command} failed with error:")
    #     print(output)

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)


    TEXT_PROMPT = word_mask
    word_mask = word_mask.split(',')  # [objec]
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image.save('input.jpg')
    image_source, image = load_image('input.jpg')

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    annotated_frame, np_arr = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    return np_arr
def ensure_rgb(image):
    """
    Ensure that the input image is in the RGB color format.

    Parameters:
    - image: The input image as a NumPy array (OpenCV format).

    Returns:
    - rgb_image: The image in the RGB color format.
    """
    # Check if the input image is already in RGB format
    if len(image.shape) == 3 and image.shape[2] == 3:
        # The image is already in RGB format
        rgb_image = image
    else:
        # Convert the image to RGB format (assuming it's in BGR format)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return rgb_image
def predict_image(image):
    """
        Predict an object's bounding box and perform necessary actions.

        Args:
            image (np.ndarray): Input image.
            word_mask (str): Object to detect.

        Returns:
            None
        """
    word_mask = "shoes"
    image = three_channels_check(image)
    h, w = 0, 0
    if image.shape[0] > 1000 or image.shape[1] > 1000:
        if image.shape[0] > 1000:
            if image.shape[0] > 2000:
                height, width, _ = image.shape
                image = resize_image(image, 1024, width)
            else:
                h = 1
                height, width, _ = image.shape
                image = resize_image(image, 512, width)
        if image.shape[1] > 1000:
            if image.shape[1] > 2000:
                height, width, _ = image.shape
                image = resize_image(image, height, 1024)
            else:
                w = 1
                height, width, _ = image.shape
                image = resize_image(image, height, 512)

        image = Image.fromarray(image)
    else:
        if image.shape[0] > 512 and image.shape[1] > 512:
            if image.shape[0] > image.shape[1]:
                image = resize_image(image, 512, image.shape[1])
            else:
                image = resize_image(image, image.shape[0], 512)
            image = Image.fromarray(image)
        else:
            image = Image.fromarray(image)

    import os
    HOME = os.getcwd()
    # load_and_download_weights(HOME)
    boxes = detect_and_create_bounding_box(image, word_mask, HOME)
    if len(boxes) > 1:
        boxes = merge_bounding_boxes(boxes)
    else:
        boxes = boxes[0]
    # %cd
    # {HOME}

    image = np.array(image)

    x_min, y_min, x_max, y_max = boxes[0], boxes[1], boxes[2], boxes[3]
    resultant, prompt = crop_and_draw_bounding_box(image, x_min, y_min, x_max, y_max, h, w)
    rgb_image = ensure_rgb(resultant)
    return rgb_image

if __name__ == "__main__":
    image = cv2.imread("test2.jpg")
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    word_mask = "shoes"
    res=predict_image(image)
    plt.imshow(res)
    plt.axis('off')
    plt.show()