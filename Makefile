check-scripts:
	shellcheck *.sh

lfs:
	git lfs track "cs224n-natural-language-processing-with-deep-learning/2023/assignments/a3/code/data/**"
	git lfs track "cs224n-natural-language-processing-with-deep-learning/2023/assignments/a4/code/zh_en_data/**"

export:
	conda env export > environment.yml

environment:
	conda env create -f environment.yml
