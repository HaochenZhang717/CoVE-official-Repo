# CoVE-official-Repo

Official implementation for ACL 2025 paper "CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems"

![Figure 1](https://github.com/HaochenZhang717/CoVE-official-Repo/raw/main/figure-1.png)


### Performance Comparison

| Datasets | Metric | GRU4Rec | BERT4Rec | Caser | SASRec | P5 | HGN | S³-Rec | FDSA | TIGER | CoVE |
|----------|--------|---------|----------|-------|--------|----|-----|--------|------|-------|------|
| **Beauty** | NG5 | 0.0099 | 0.0124 | 0.0131 | 0.0249 | 0.0107 | 0.0206 | 0.0244 | 0.0163 | <u>0.0321</u> | **0.0498** |
| | NG10 | 0.0137 | 0.0170 | 0.0176 | 0.0318 | 0.0136 | 0.0266 | 0.0327 | 0.0208 | <u>0.0384</u> | **0.0593** |
| | HR5 | 0.0164 | 0.0203 | 0.0205 | 0.0387 | 0.0163 | 0.0325 | 0.0387 | 0.0267 | <u>0.0454</u> | **0.0714** |
| | HR10 | 0.0283 | 0.0347 | 0.0347 | 0.0605 | 0.0254 | 0.0512 | 0.0647 | 0.0407 | <u>0.0648</u> | **0.1009** |
| **Toys** | NG5 | 0.0059 | 0.0071 | 0.0107 | 0.0306 | 0.0050 | 0.0221 | 0.0294 | 0.0140 | <u>0.0371</u> | **0.0509** |
| | NG10 | 0.0084 | 0.0099 | 0.0141 | 0.0374 | 0.0066 | 0.0277 | 0.0376 | 0.0189 | <u>0.0432</u> | **0.0595** |
| | HR5 | 0.0097 | 0.0116 | 0.0166 | 0.0463 | 0.0070 | 0.0321 | 0.0443 | 0.0228 | <u>0.0521</u> | **0.0719** |
| | HR10 | 0.0176 | 0.0203 | 0.0270 | 0.0675 | 0.0121 | 0.0497 | 0.0700 | 0.0381 | <u>0.0712</u> | **0.0986** |
| **Sports** | NG5 | 0.0086 | 0.0075 | 0.0072 | 0.0192 | 0.0041 | 0.0120 | <u>0.0204</u> | 0.0156 | 0.0181 | **0.0296** |
| | NG10 | 0.0110 | 0.0099 | 0.0097 | <u>0.0249</u> | 0.0052 | 0.0159 | 0.0240 | 0.0156 | 0.0225 | **0.0359** |
| | HR5 | 0.0129 | 0.0115 | 0.0116 | 0.0233 | 0.0061 | 0.0189 | 0.0251 | 0.0182 | <u>0.0264</u> | **0.0428** |
| | HR10 | 0.0204 | 0.0191 | 0.0194 | 0.0350 | 0.0095 | 0.0313 | 0.0385 | 0.0288 | <u>0.0400</u> | **0.0624** |

*Performance comparison of different recommendation models across three datasets: Beauty, Toys and Games, and Sports and Outdoors. The results of CoVE shown in the table use a compression rate of 2. The best metrics are always achieved by using CoVE and are all in bold font in this table, and the best metric achieved by prior works is underlined.*


## Download data

Because of the dataset's size, it cannot be uploaded to the Repo. Please download data from here:  
[https://drive.google.com/drive/folders/1h_swkdw4Evp7X4iNaOYczlf_vvCd_Px6?usp=share_link](https://drive.google.com/drive/folders/1h_swkdw4Evp7X4iNaOYczlf_vvCd_Px6?usp=share_link)

## Reimplement experiment

Please run `run_all_experiment.sh` for reimplementation.
