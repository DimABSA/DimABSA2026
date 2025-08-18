<figure>
  <img src="https://github.com/DimABSA/DimABSA2026/blob/main/banner.png" width="100%">
</figure>
<!-- <figure>
  <img src="./assets/banner.png" width="100%">
</figure> -->

# Content
- [Overview](#overview)
- [Task Description](#task-description)
    - [Subtask 1](#subtask-1-dimensional-aspect-sentiment-regression-dimasr)
    - [Subtask 2](#subtask-2-dimensional-aspect-sentiment-triplet-extraction-dimaste) 
    - [Subtask 3](#subtask-3-dimensional-aspect-sentiment-quad-prediction-dimasqp)
- [Datasets](#datasets)
- [Full List of Aspect Categories](#full-list-of-aspect-categories)
- [Important Dates and Task Phases](#important-dates-and-task-phases)
- [Resources](#resources)
- [References](#references)


[Join the Google Group for the task](https://groups.google.com/g/dimabsa-participants)| [Join Discord](#)  | [Create an Issue](#) | [Contact Us](mailto:dimabsa-organizers@googlegroups.com) | [Download Dataset](#) | [How to Participate](#)|

---

# Overview

Aspect-Based Sentiment Analysis (ABSA) is the task of identifying aspect terms in text and determining their associated sentiment polarity. However, current ABSA research predominantly adopts a coarse-grained, categorical sentiment representation (e.g., positive, negative, or neutral). This approach stands in contrast to long-established theories in psychology and affective science (Russell, 1980; 2003), where sentiment is represented along fine-grained, real-valued dimensions of **valence** (ranging from negative to positive) and **arousal** (from sluggish to excited), as illustrated in Fig. 1.

This valence-arousal (VA) representation has inspired the rise of dimensional sentiment analysis as an emerging research paradigm (Mohammad, 2018; Lee et al., 2022, 2024; Muhammad et al., 2025), enabling more nuanced distinctions in emotional expression and supporting a broader range of applications.


<p align="center">
  <img src="https://github.com/DimABSA/DimABSA2026/blob/main/VA.png" width="500"><br>
  Fig. 1. Two-dimensional valence-arousal space (Yu et al., 2016).
</p>


To bridge this gap, this task introduces **Dimensional ABSA (DimABSA)**, a task that integrates dimensional sentiment analysis into the traditional ABSA framework and is defined as:

> Given a textual instance, participants are expected to predict real-valued **valence** and **arousal** scores associated with relevant aspects. The proposed DimABSA task has the following features


- **Fine-grained sentiment modelling**: Replaces discrete sentiment labels with continuous VA scores, offering more nuanced emotional representation.
- **Multidomain coverage**: Constructs datasets from four diverse application areas, **customer reviews**,**moview reviews** **financial reports**, and **stance detection**, going beyond the focus on consumer reviews on traditional ABSA.
- **Multilingual benchmark**: Covers languages across five continents, including both high-resource and low-resource languages. Supported languages include: *Hausa, Igbo, Kinyarwanda, Swahili, Twi, Chinese, English, German, Japanese, Brazilian Portuguese, Russian, Ukrainian, and Tatar*.


# Task Description
<!--  
This shared task introduces **dimensional sentiment representation** into three established Aspect-Based Sentiment Analysis (ABSA) subtasks: **Aspect Sentiment Classification (ASC)**, **Aspect Sentiment Triplet Extraction (ASTE)**, and **Aspect Sentiment Quad Prediction (ASQP)**. Traditionally, these subtasks involve the identification of aspect terms, aspect categories, opinion terms, and the assignment of discrete sentiment polarity labels (e.g., positive, negative, neutral) within a given sentence.

To enable a more fine-grained and psychologically grounded understanding of sentiment, we replace categorical polarity labels with continuous **valence-arousal (VA)** scores. These scores provide a two-dimensional representation of affective meaning, capturing both the **valence** (i.e., the positivity or negativity of the sentiment) and **arousal** (i.e., the degree of emotional intensity).
-->
 We define three subtasks, each corresponding to a traditional ABSA task but adapted to the dimensional sentiment paradigm:

- **Dimensional Aspect Sentiment Regression (DimASR)**: predicts VA scores for aspect terms in text.
- **Dimensional Aspect Sentiment Triplet Extraction (DimASTE)**: extracts (aspect term, opinion term, VA score) triplets.
- **Dimensional Aspect Sentiment Quad Prediction (DimASQP)**: extracts (aspect category, aspect term, opinion term, VA score) quads.

The elements to be predicted in the new subtasks are described as follows.

- **Aspect Term**: A word or phrase indicating an opinion target, such as *appetizer*, *waiter*, *battery*, or *screen*.
- **Aspect Category**: An abstract or predefined category to which an aspect term belongs. It follows the format *Entity#Attribute*, where the *Entity* (e.g., `FOOD`, `SERVICE`) and *Attribute* (e.g., `PRICES`, `QUALITY`) are selected from predefined lists (Pontiki et al., 2015). For all valid combinations, see the [full list of aspect categories](#full-list-of-aspect-categories).
- **Opinion Term**: A sentiment-bearing word or phrase associated with a specific aspect term, such as *great*, *terrible*, or *satisfactory*.
- **Valence-Arousal (VA)**: A pair of real-valued scores, each ranging from **1.00 to 9.00**, rounded to two decimal places.  
    - **Valence (V)**: Measures the degree of positivity or negativity.  
    - **Arousal (A)**: Measures the intensity of emotion.  
    A score of **1.00** indicates extremely negative valence or very low arousal, **9.00** indicates extremely positive valence or very high arousal, and **5.00** represents a neutral valence or medium arousal.

Participants may choose to participate in one or more of these subtasks, depending on their research interest or application focus.

<!--  

We introduce dimensional sentiment representation into three traditional ABSA subtasks: Aspect Sentiment Classification (ASC), Aspect Sentiment Triplet Extraction (ASTE), and Aspect Sentiment Quad Prediction (ASQP), which involve the analysis of aspect terms, aspect categories, opinion terms, and sentiment polarity in given sentences. By replacing sentiment polarity with valence-arousal scores, we propose three new corresponding dimensional subtasks: Dimensional Aspect Sentiment Regression (DimASR), Dimensional Aspect Sentiment Triplet Extraction (DimASTE), and Dimensional Aspect Sentiment Quad Prediction (DimASQP). The resulting elements to be predicted in the new subtasks are described as follows. Among them, only the Valence-Arousal (VA) element is newly introduced; the others are identical to those in traditional ABSA.

- Aspect Term: refers to a word or phrase indicating an opinion target, such as *appetizer*, *waiter*, *battery*, or *screen*.
- Aspect Category: refers to the abstract or predefined category to which an aspect term belongs. It is denoted in the format *Entity#Attribute*, where the Entity (e.g., FOOD, SERVICE) and the Attribute (e.g., PRICES, QUALITY) are both selected from predefined lists (Pontiki et al., 2015). For all valid combinations, see the [full list of aspect categories](#full-list-of-aspect-categories).
- Opinion Term: refers to a sentiment-bearing word or phrase associated with a specific aspect term, such as *great*, *terrible*, or *satisfactory*.
- Valence-Arousal (VA): Both valence (V) and arousal (A) are real numbers ranging from 1.00 to 9.00, rounded to two decimal places. A score of 1.00 indicates extremely negative valence or very low arousal; 9.00 indicates extremely positive valence or very high arousal; and 5.00 represents neutral valence or medium arousal.

Different subtasks involve different combinations of the above elements. Participants may choose to participate in one or more of these subtasks.
-->
## Subtask 1: Dimensional Aspect Sentiment Regression (DimASR)

Given a textual instance and one or more target aspects, predict a real-valued **valence-arousal (VA)** score for each aspect. 
The input is in JSON Lines format and includes the following fields.
- "ID" – A unique identifier for the instance.
- "Text" – A sentence or paragraph expressing subjective opinions.
- "Aspect" – A list of one or more target aspects mentioned in the text.

The output should be in JSON Lines format and include the following fields. All textual outputs are **case-sensitive**.

- "ID" – Should match the input ID.
- "Aspect_VA" – A list of pairs, where each pair contains the following fields.
    - "Aspect" – Should be identical in content, case, and order to the Aspect list in the input.
    - "VA" – The valence-arousal score as a string in V#A format, with each value ranging from 1.00 to 9.00 and rounded to two decimal places.

Below are some examples from domains included in this subtask, such as restaurant, laptop, and hotel reviews, and financial reports, and environmental stance detection.


- ### Restaurant
  Input: 
  ```json
  {
      "ID": "R001",
      "Text": "average to good thai food, but terrible delivery.",
      "Aspect": [
          "thai food",
          "delivery"
      ]
  }
  ```
  
  Output:
  ```json
  {
      "ID": "R001",
      "Aspect_VA":[
          {
              "Aspect": "thai food",
              "VA": "6.75#6.38"
          },
          {
              "Aspect": "delivery",
              "VA": "2.88#6.62"
          }
      ]
  }
  ```
- ### Laptop
  Input:
  ```json
  {
      "ID": "L001",
      "Text": "i am extremely happy with this laptop.",
      "Aspect": [
          "laptop"
      ]
  }
  ```
  Output:
  ```json
  {
      "ID": "L001",
      "Aspect_VA":[
          {
              "Aspect": "laptop",
              "VA": "8.12#8.25"
          }
      ]
  }
  ```

- ### Hotel

  Input:
  ```json
  {
      "ID": "H001",
      "Text": "Check-in was smooth, and the room was perfectly clean.",
      "Aspect": [
          "Check-in",
          "room"
      ]
  }
  ```
  Output:
  ```json
  {
      "ID": "H001",
      "Aspect_VA":[
          {
              "Aspect": "Check-in",
              "VA": "6.33#5.25"
          },
          {
              "Aspect": "room",
              "VA": "7.88#8.33"
          }
      ]
  }
  ```
- ### Finance
  Input:
  ```json
  
  {
      "ID": "F001",
      "Text": "The pandemic led to a record low in net income.",
      "Aspect": [
          "net income"
      ]
  }
  ```
  Output:
  ```json
  {
      "ID": "F001",
      "Aspect_VA":[
          {
              "Aspect": "net income",
              "VA": "2.14#7.67"
          }
      ]
  }
  ```
- ### Stance
  Input:
  ```json
  
  {
      "ID": "S001",
      "Text": "We must walk door to door in our communities even as ws demand change form the top.",
      "Aspect": [
          "communities"
      ]
  }
  ```
  Output:
  ```json
  {
      "ID": "S001",
      "Aspect_VA":[
  
          {
              "Aspect": "communities",
              "VA": "6.83#7.30"
          }
      ]
  }
  ```


## Subtask 2: Dimensional Aspect Sentiment Triplet Extraction (DimASTE)
Given a textual instance, extract all **(A, O, VA)** triplets, where A denotes an aspect term, O an opinion term, and VA a valence-arousal score. 
The input is in JSON Lines format and includes the following fields.
- "ID" – A unique identifier for the instance.
- "Text" – A sentence or paragraph expressing subjective opinions. 

The output should be in JSON Lines format and include the following fields. All textual outputs are **case-sensitive**.

- "ID" – Should match the input ID.
- "Triplet" – A list of extracted triplets, where each triplet contains the following fields.
    - "Aspect" – The aspect term (string), which should retain the same case as in the input text.
    - "Opinion" – The opinion term (string), which should retain the same case as in the input text.
    - "VA" – The valence-arousal score as a string in V#A format, with each value ranging from 1.00 to 9.00 and rounded to two decimal places.
Below are some examples from domains included in this subtask, such as restaurant, laptop, and hotel reviews, and financial reports.

- ### Restaurant
  Input:
  ```json
  
  {
      "ID": "R001",
      "Text": "average to good thai food, but terrible delivery."
  }
  ```
  Output:
  ```json
  {
      "ID": "R001",
      "Triplet":[
          {
              "Aspect": "thai food",
              "Opinion": "average to good",
              "VA": "6.75#6.38"
          },
          {
              "Aspect": "delivery",
              "Opinion": "terrible",
              "VA": "2.88#6.62"
          }
      ]
  }
  ```
- ### Laptop
  Input:
  ```json
  
  {
      "ID": "L001",
      "Text": "i am extremely happy with this laptop.",
      "Aspect": [
          "laptop"
      ]
  }
  ```
  Output:
  ```json
  {
      "ID": "L001",
      "Triplet":[
          {
              "Aspect": "laptop",
              "Opinion": "xtremely happy",
              "VA": "8.12#8.25"
          }
      ]
  }
  ```

- ### Hotel

  Input:
  ```json
  {
      "ID": "H001",
      "Text": "Check-in was smooth, and the room was perfectly clean."
  }
  ```
  Output:
  ```json
  {
      "ID": "H001",
      "Triplet":[
          {
              "Aspect": "Check-in",
              "Opinion": "smooth",
              "VA": "6.33#5.25"
          },
          {
              "Aspect": "room",
              "Opinion": "perfectly clean",
              "VA": "7.88#8.33"
          }
      ]
  }
  ```
- ### Finance
  Input:
  ```json
  
  {
      "ID": "F001",
      "Text": "The pandemic led to a record low in net income."
  }
  ```
  Output:
  ```json
  {
      "ID": "F001",
      "Triplet":[
          {
              "Aspect": "net income",
              "Opinion": "record low",
              "VA": "2.14#7.67"
          }
      ]
  }
  ```
## Subtask 3: Dimensional Aspect Sentiment Quad Prediction (DimASQP)
Given a textual instance, extract all **(A, C, O, VA)** quadruplets, where A denotes an aspect term, C an aspect category, O an opinion term, and VA a valence-arousal score. This task is an extension of Subtask 2 (triplet extraction), with the addition of the aspect category element.
The input is in JSON Lines format and includes the following fields:
- "ID" – A unique identifier for the instance.
- "Text" – A sentence or paragraph expressing subjective opinions.

The output should be in JSON Lines format and include the following fields. All textual outputs are **case-sensitive**.
- "ID" – Should match the input ID.
- "Quadruplet" – A list of extracted quadruplets, where each quadruplet includes the following fields.
    - "Aspect" – The aspect term (string), which should retain the same case as in the input text.
    - "Category" – The aspect category (string), formatted as ENTITY#ATTRIBUTE and written in UPPERCASE. For all valid combinations, see the [full list of aspect categories](#full-list-of-aspect-categories).
    - "Opinion" – The opinion term (string), which should retain the same case as in the input text.
    - "VA" – The valence-arousal score as a string in V#A format, with each value ranging from 1.00 to 9.00 and rounded to two decimal places.

Below are some examples from domains included in this subtask, such as restaurant, laptop, and hotel reviews, and financial reports.

- ### Restaurant
  Input:
  ```json
  {
      "ID": "R001",
      "Text": "average to good thai food, but terrible delivery."
  }
  ```
  Output:
  ```json
  {
      "ID": "R001",
      "Quadruplet":[
          {
              "Aspect": "thai food",
              "Category": "FOOD#QUALITY",
              "Opinion": "average to good",
              "VA": "6.75#6.38"
          },
          {
              "Aspect": "delivery",
              "Category": "SERVICE#GENERAL",
              "Opinion": "terrible",
              "VA": "2.88#6.62"
          }
      ]
  }
  ```
- ### Laptop
  Input:
  ```json
  {
      "ID": "L001",
      "Text": "i am extremely happy with this laptop.",
      "Aspect": [
          "laptop"
      ]
  }
  ```
  Output:
  ```json
  {
      "ID": "L001",
      "Quadruplet":[
          {
              "Aspect": "laptop",
              "Category": "LAPTOP#GENERAL",
              "Opinion": "xtremely happy",
              "VA": "8.12#8.25"
          }
      ]
  }
  ```

- ### Hotel
  Input:
  ```json
  {
      "ID": "H001",
      "Text": "Check-in was smooth, and the room was perfectly clean."
  }
  ```
  Output:
  ```json
  {
      "ID": "H001",
      "Quadruplet":[
          {
              "Aspect": "Check-in",
              "Category": "SERVICE#QUALITY",
              "Opinion": "smooth",
              "VA": "6.33#5.25"
          },
          {
              "Aspect": "room",
              "Category": "ROOMS#CLEANLINESS",
              "Opinion": "perfectly clean",
              "VA": "7.88#8.33"
          }
      ]
  }
  ```
- ### Finance
  Input:
  ```json
  {
      "ID": "F001",
      "Text": "The pandemic led to a record low in net income."
  }
  ```
  Output:
  ```json
  {
      "ID": "F001",
      "Quadruplet":[
          {
              "Aspect": "net income",
              "Category": "NULL#PROFIT"
              "Opinion": "record low",
              "VA": "2.14#7.67"
          }
      ]
  }
  ```
# Datasets
| No. | Language | Code | Subtask 1<br>DimASR | Subtask 2<br>DimASTE | Subtask 3<br>DimASQP |
|:-----:|:----------:|:------:|:------------------:|:-------------------:|:------------------:|
| 1 | Chinese | ZHO | Restaurant<br>Laptop |  Restaurant<br>Laptop | Restaurant<br>Laptop |
| 2 | English | ENG | Restaurant<br>Laptop<br>Stance | Restaurant<br>Laptop<br>Stance | Restaurant<br>Laptop |
| 3 | German | DEU | Stance | Stance |  |
| 4 | Hausa | HAU |  |  |  |
| 5 | Igbo | IBO |  |  |  |
| 6 | Japanses | JPN | Hotel<br>Finance | Hotel<br>Finance | Hotel<br>Finance |
| 7 | Kinyarwanda| KIN |  |  |  |
| 8 | Portuguese <br> (Brazilian) | PTBR |  |  |  |
| 9 | Russian | RUS | Restaurant | Restaurant |  |
| 10 | Swahili | SWA |  |  |  |
| 11 | Tatar | TAT | Restaurant | Restaurant |  |
| 12 | Twi | TWI |  |  |  |
| 13 | Ukrainian | UKR | Restaurant | Restaurant |  |


# Evaluation

The performance of the submitted systems will be evaluated based on the following metrics:


- <b> Subtask 1: Dimensional Aspect Sentiment Regression (DimASR) </b> 
   TBD
- <b>Subtask 2: Dimensional Aspect Sentiment Triplet Extraction (DimASTE) </b> 
TBD
- <b>Subtask 3: Dimensional Aspect Sentiment Quad Prediction (DimASQP)</b> 
TBD
- For details about the evaluation script and the submission file format checker, check this [guide](#).


# Important Dates and Task Phases

| Description                   | Deadline                                        |
|-------------------------------|------------------------------------------------|
| Sample Data Ready             | 15 July 2025                                    |
| Training Data Ready           | 1 September 2025                                |
| Evaluation Start              | 10 January 2026                                 |
| Evaluation End                | 31 January 2026                                 |
| System Description Paper Due  | February 2026                                   |
| Notification to Authors       | March 2026                                      |
| Camera Ready Due              | April 2026                                      |
| SemEval Workshop 2026         | (co-located with ACL 2026)                      |



# Full List of Aspect Categories 
> from [SemEval-2016 Task 5](https://aclanthology.org/S16-1002.pdf)

## Restaurant
|Entity Labels|
|-------------|
|RESTAURANT, FOOD, DRINKS, AMBIENCE, SERVICE, LOCATION|
|**Attribute Labels**|
|GENERAL, PRICES, QUALITY, STYLE_OPTIONS, MISCELLANEOUS|

## Laptop
|Entity Labels|
|-------------|
|LAPTOP, DISPLAY, KEYBOARD, MOUSE, MOTHERBOARD, CPU, FANS_COOLING, PORTS, MEMORY, POWER_SUPPLY, OPTICAL_DRIVES, BATTERY, GRAPHICS, HARD_DISK, MULTIMEDIA_DEVICES, HARDWARE, SOFTWARE, OS, WARRANTY, SHIPPING, SUPPORT, COMPANY|
|**Attribute Labels**|
|GENERAL, PRICE, QUALITY, DESIGN_FEATURES, OPERATION_PERFORMANCE, USABILITY, PORTABILITY, CONNECTIVITY, MISCELLANEOUS|

## Hotel
|Entity Labels|
|-------------|
|HOTEL, ROOMS, FACILITIES, ROOM_AMENITIES, SERVICE, LOCATION, FOOD_DRINKS|
|**Attribute Labels**|
|GENERAL, PRICE, COMFORT, CLEANLINESS, QUALITY, DESIGN_FEATURES, STYLE_OPTIONS, MISCELLANEOUS|

## Finance
|Entity Labels|
|-------------|
|MARKET, COMPANY, BUSINESS, PRODUCT|
|**Attribute Labels**|
|GENERAL, SALES, PROFIT, AMOUNT, PRICE, COST|
> https://github.com/chakki-works/chABSA-dataset



# Resources

1. Previous shared tasks on sentiment regression
- [SemEval-2025 Task 11](https://github.com/emotion-analysis-project/SemEval2025-task11) (Track B)
- [SIGHAN-2024 Shared Tasks](https://github.com/NYCU-NLP/SIGHAN2024-dimABSA) (Chinese)
- [SemEval-2018 Task 1](https://competitions.codalab.org/competitions/17751) (EI-reg, V-reg)
- [WASSA-2017 Shared Task](https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html)

2.	Dimensional Sentiment Corpora
- [EmoBank](https://github.com/JULIELab/EmoBank)
- [Chinese Emobank](http://nlp.innobic.yzu.edu.tw/resources/ChineseEmoBank.html)
- [Facebook Posts](https://github.com/wwbp/additional_data_sets/tree/master/valence_arousal)
- [COMETA (German)](https://link.springer.com/article/10.3758/s13428-019-01300-7)

# References

Sven Buechel and Udo Hahn. 2017. EmoBank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis. In *Proc. of EACL-17*, pages 578-585.

Francesca M. M. Citron, Mollie Lee, and Nora Michaelis. 2020. Affective and psycholinguistic norms for German conceptual metaphors (COMETA). *Behavior Research Methods*, 52(3):1056-1072.

Will E. Hipson and Saif M. Mohammad. 2021. Emotion dynamics in movie dialogues. *PLOS ONE*, 16(9):e0256153.

Lung-Hao Lee, Jian-Hong Li, and Liang-Chih Yu. 2022. Chinese EmoBank: Building Valence-Arousal Resources for Dimensional Sentiment Analysis. *ACM Transactions on Asian and Low-Resource Language Information Processing*, 21(4):65.

Lung-Hao Lee, Liang-Chih Yu, Suge Wang and Jian Liao. Overview of the SIGHAN 2024 shared task for Chinese dimensional aspect-based sentiment analysis. In *Proc. of SIGHAN-24*, pages 165-174.

Zhiwei Liu, Tianlin Zhang, Kailai Yang, Paul Thompson, Zeping Yu, Sophia Ananiadou. 2024. Emotion detection for misinformation: A review. Information Fusion, 107:102300.
Saif M. Mohammad. 2018. Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words. *In Proc. of ACL-18*, pages 174-184.

Saif M. Mohammad, Felipe Bravo-Marquez, Mohammad Salameh, and Svetlana Kiritchenko. 2018. SemEval-2018 Task 1: Affect in Tweets. *In Proc. of SemEval-18*, pages 1-17.

Shamsuddeen Hassan Muhammad, Nedjma Ousidhoum, Idris Abdulmumin, Seid Muhie Yimam, Jan Philip Wahle, Terry Ruas, Meriem Beloucif, Christine De Kock, Tadesse Destaw Belay, Ibrahim Said Ahmad, Nirmal Surange, Daniela Teodorescu, David Ifeoluwa Adelani, Alham Fikri Aji, Felermino Ali, Vladimir Araujo, Abinew Ali Ayele, Oana Ignat, Alexander Panchenko, Yi Zhou, and Saif M. Mohammad. 2025. SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection. In *Proc. of SemEval-25*.

Maria Pontiki, Dimitrios Galanis, Haris Papageorgiou, Ion Androutsopoulos, Suresh Manandhar, Mohammad AL-Smadi, Mahmoud AI-Ayyoub, Yanyan Zhao, Bing Qin, Orphee De Clercq, Veronique Hoste, Marianna Apidianaki, Xavier Tannier, Natalia Loukachevitch. Evgeny Kotelnikov, Nuria Bel, Salud Maria Jimenez-Zafra and Gulsen Eryigit. 2016. SemEval-2016 Task 5: Aspect Based Sentiment Analysis. In *Proc. of SemEval-16*, pages 19-30.

Maria Pontiki, Dimitrios Galanis, Haris Papageorgiou, Suresh Manandhar and Ion Androutsopoulos. SemEval-2015 Task 12: Aspect Based Sentiment Analysis. In *Proc. of SemEval-15*, pages 486-495.

Maria Pontiki, Dimitrios Galanis, John Pavlopoulos, Haris Papageorgiou, Ion Androutsopoulos and Suresh Manandhar. 2014. SemEval-2014 Task 4: Aspect Based Sentiment Analysis. In *Proc. of SemEval-14*, pages 27-35. 

James A Russel. 1980. A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6):1161-1178.

James A Russel. 2003. Core affect and the psychological construction of emotion. *Psychological Review*, 110(1):145172.

Shuvam Shiwakoti, Surendrabikram Thapa, Kritesh Rauniyar, Akshyat Shah, Aashish Bhandari and Usman Naseem. 2024. Analyzing the Dynamics of Climate Change Discourse on Twitter: A New Annotated Corpus and Multi-Aspect Classification. In *Proceedings of LREC/COLING-24*, pages 984–994.

Daniela Teodorescu, Tiffany Cheng, Alona Fyshe and Saif M. Mohammad. 2023. Language and Mental Health: Measures of Emotion Dynamics from Text as Linguistic Biosocial Markers. In *Proc. of EMNLP-23*, pages 3117-3133.

Apoorva Upadhyaya, Marco Fisichella and Wolfgang Nejdl. Toxicity, Morality, and Speech Act Guided Stance Detection. In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 4464–4478.

Zhiyuan Wen, Jiannong Cao, Jiaxing Shen, Ruosong Yang, Shuaiqi Liu and Maosong Sun. 2024. Personality-affected Emotion Generation in Dialog Systems. *ACM Trans. Information Systems*, 42(5):134.

Liang-Chih Yu, Lung-Hao Lee, Shuai Hao, Jin Wang, Yunchao He, Jun Hu, K Robert Lai, Xuejie Zhang. 2016. Building Chinese Affective Resources in Valence-Arousal Dimensions. In *Proc. of NAACL-16*, pages 540-545.

Wenxuan Zhang, Xin Li, Yang Deng, Lidong Bing and Wai Lam. 2023. A Survey on Aspect-Based Sentiment Analysis: Tasks, Methods, and Challenges. *IEEE Trans. Knowledge and Data Engineering*, 35(11):11019-11038.

# Organizers
[Liang-Chih Yu](http://nlp.innobic.yzu.edu.tw/~lcyu/), [Shamsuddeen Hassan Muhammad](https://shmuhammadd.github.io/), [Lung-Hao Lee](https://lunghao.weebly.com/), [Jin Wang](http://www.ise.ynu.edu.cn/teacher/973), [Jan Philip Wahle](https://jpwahle.com/), [Terry Ruas](https://terryruas.com/), [Kai-Wei Chang](https://web.cs.ucla.edu/~kwchang/), [Saif M. Mohammad](https://www.saifmohammad.com/)

