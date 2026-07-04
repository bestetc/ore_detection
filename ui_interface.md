UI interface demand.

Main goal: 
1. Check the different models prediction.
2. Update the mask for the active learning loop.
3. Working with large images (500M pixels).

UI:
On top should be setting:
	- model current model. could be changed.
	- image name/address. could be changed. add the drag and drop image.

Start status:
Main field with no image loading.
Start scale - full image should be place inside a window.
No model selected.


Main field:
- image in ore_detection/datasets/set_1/masks_human/train/train_01.jpg format.
	- three images in a row. raw image, raw_image + mask, mask only.
	- change the image. load new image and make prediction.
	- change the using model.
	- mask should be converted from one-hot model output to the different colors.
	- add description color - class type (ore type). use description from ./docs/label_mapping.md
	- label_mapping.md do not contain talc class. add it. color for talc mask should be white.
	
- load a new raw images. after loading calculate the mask by current model.
- change the current model.
	
Models:
- choise one of next models:
	- binary segmentation. 
	- ore segmentation.
	- intergrowth segmentation. (current not ready).
- show the `Color → class description` only for current model. The normal ore, hard ore, and talc is avialable only for intergrowth segmentation model.

	
Metrics:
show metrics calculated on the images.
- part of hard ore mask square.
- part of normal ore mask square.
- part of talc ore mask square.

Instrument:
	View (do not change or creating new artifacts): 
	At every moment in time, the system shows exactly the same slice of the image.
	- scaling images. 
		- all images should be scaled together only.
		- images could be scale as `+` and `-` button.
	- crop images.
		- crop area that shown. do not 
		- push `crop` button and choise the are on any images that should be cropped.
	- return to full view.
	Active learning tools:
	- change the mask by classes by brush.
		- mask class is currently being edited.
		- choice from all model classes.
		- add the talc class, normal ore class, hard ore class.
		- change size of brush.
		Add the mask editing tools:
		- add the brush to add or remove masked region.
		- show the brush borders by write color.
		- choise the class that are current added. If I choice class "backgroud" it mean that all brushed pixel now are move to backgroud class.
		- update the mask view in live.
	Save full mask tensor as one-hot mask to ./data_work folder. Name of file should be correspond with source image name.
	Result: torch.tensor for training loop.
	
	Talc mask creation.
	To bottom of UI add a two plot. (under main three plot)
		- Histogram of raw image by Value of HSV.
		- Histogram of raw image by R+G+B.
	Add the mask creation tools:
		- slider for the selected metric (Value of HSV or R+G+B)
		- talc mask is all pixel that has value less metric.
		- add the brush to add or remove masked region.
		- update the mask in live.

	
	
