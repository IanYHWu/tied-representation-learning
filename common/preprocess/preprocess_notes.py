"""

Preprocess idea:

Create a base class for a preprocessed dataset that handles
language directions and dataloaders. Then create a seperate
script for each dataset that handles the preprocessing for it
and returns an instance of the class. When dataset name is
passed the relevant function is called for the dataset. Then
to add a new dataset only the new script needs to be written.

Divide preprocess into multiple modules:
	preprocess main:
		holds the code for loading a dataset that is called by
		other scripts
	preprocess utils:
		holds common preprocessing code such as training code
		for tokenizers
	for each dataset:
		a script to load that specific dataset

"""