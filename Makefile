.PHONY: conda pip clean

conda:
	conda env create -f environment.yaml

update:
	conda env update --file environment.yaml --prune

pip:
	pip install nibabel

clean:
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints
	conda env remove -n rec-xml-to-nii
