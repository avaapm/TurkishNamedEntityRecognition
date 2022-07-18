# Named entity recognition in Turkish: A comparative study with detailed error analysis

This repository contains the implementation codes, results of each evaluation repeat, and the details of statistical test results in the paper "Named entity recognition in Turkish: A comparative study with detailed error analysis". This study provides a comparative analysis on the performances of the state-of-the-art approaches for Turkish named entity recognition using existing datasets with varying domains. The study includes a detailed error analysis that examines both quantitative (entity types, varying entity lengths, and changing word orders) and qualitative (ambiguous entities and noisy texts) factors that can affect the model performance.

## Evaluation Results

In "Evaluation Results" folder, evaluation results of 10 random initalizations of each model for each dataset are presented.

| Repeat  | F1 | Precision | Recall |
| ------------- | ------------- | ------------- | ------------- |
| 1-Repeat | 0.8762163224236367 | 0.8744340791601422 | 0.8803910293271996 |
| 2-Repeat | 0.8722907171938192 | 0.8668121509523580 | 0.8803910293271996 |
| 3-Repeat | 0.8800237284845157 | 0.8736443760514383 | 0.8867165037377803 |
| ... | ... | ... | ... |
| Avg. 10-Repeat | 0.87719 | 0.87199 | 0.88413 |



## Citation
If you make use of this dataset, please cite following paper.

```bibtex
@misc{ozcelik_and_toraman2022ner,
      title={Named entity recognition in Turkish: A comparative study with detailed error analysis}, 
      author={Oguzhan Ozcelik and Cagri Toraman},
      year={2022},
      url={https://github.com/avaapm/TurkishNamedEntityRecognition/}
}

```

## Statistical test results

Statistical test results, indicated in Table 3 and 4, are provided in our paper.
