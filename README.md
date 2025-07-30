# NEURAL-STYLE-TRANSFER

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: AKULA GOWRI SHANKAR

*INTERN ID*: CT04DH833

*DOMAIN*: AI(ARTIFICIAL INTELLIGENCE)

*DURATION*:4 WEEKS

*MENTOR*: NEELA SANTOSH

## DESCRIPTION

1. Image Loading and Preprocessing
The load_image() function handles:

Opening an image file and converting it to RGB.

Resizing and normalizing the image to match the input format expected by VGG-19.

Converting the image into a PyTorch tensor and adding a batch dimension.

Normalization is crucial because the VGG model was trained on ImageNet with a specific mean and standard deviation.

2. Image Conversion for Display
The im_convert() function:

Converts a tensor back into a NumPy image array.

Undoes the normalization so that we can visualize the image in normal RGB color space.

Clips pixel values to be between 0 and 1 for valid image rendering.

3. Feature Extraction
The get_features() function extracts layers from the VGG model. Different layers of a CNN capture different information:

Lower layers: texture, patterns.

Deeper layers: object and shape representations.

By selecting specific layers like conv1_1, conv2_1, ..., conv4_2, we extract both style and content features.

4. Style Representation
The gram_matrix() function converts feature maps into Gram matrices, which represent texture and style by measuring correlations between filter responses.

5. Training the Target Image
The run_style_transfer() function drives the transformation. Here's what happens:

We start with a copy of the content image and make it a parameter that will be updated (this is the "target").

Using the extracted features and Gram matrices, we calculate two losses:

Content Loss: Ensures the target retains content from the original image.

Style Loss: Makes the target resemble the style of the style image.

A combined total loss is computed.

We use the Adam optimizer to iteratively update the target image to minimize the total loss.

This loop runs for 2000 steps (customizable), gradually transforming the image to balance both content and style.

6. Saving and Showing the Result
At the end, the stylized image is:

Converted from a tensor to a displayable image.

Saved as "styled_output.jpg".

Displayed using matplotlib.

## OUTPUT

![Image](https://github.com/user-attachments/assets/9d477a0e-07bd-4a6c-a5ae-d6faa1834115)
