# Clothing Recommender System
ML Camp Jeju 2017
Sally Hong
Mentor: Dylan Byeon

---

## Table of Contents
1. [Overview](#overview)
    1. [Datasets Used](#dataset)
3. [Workflow](#workflow)
    1. [Complete Workflow](#complete)
    2. [Part I : Object Detection](#p1)
    3. [Part II : Attribute Tagging](#p2)
    4. [Part III : Recommendation based on Frequency](#p3)
4. [Future Direction and Ideas](#future)
5. [Final Thoughts](#thoughts) 
6. [Sources/Acknowledgements](#sources)

---

## <a id="overview"> Overview/Motivation

During my month at Jeju ML Camp, I proposed and implemented a basic potential workflow for a clothing recommender system.

My motivation stemmed from a simple question: _"What is Fashion?"_
We have intrinsic knowledge on what kind of outfits 'work' (in other words, looks 'good')... then, how can we translate this 'knowledge' to computers?

This topic was interesting to me because **fashion is an evolution**. 
Depending on what is trending at the moment, the results can be different.

My workflow consisted of three main parts:
1. Object Detection
    * Used a Faster-RCNN implementation via Keras but object detection was limited to "top" and "bottom" clothing
2. Attribute Tagging
    * Used CNN based on the Inception V3 model (with weights provided by Google)
    * The final layer was adjusted to fit the "attribute taggings"
3. Recommendation based on Frequency
    * Note: This portion relied upon good data from part 2. However, due to the imbalanced dataset and lack of training time, I was unable to get satisfactory results. Hence, I used a "dummy" matrix as a placeholder to demonstrate the concept.

Please note that the codes/notebooks contains superfluous code and texts due to debugging and testing. Definitely needs a lot of code cleanup if such project were to be realized in a production enviornment.

### <a id="dataset"> Datasets Used

For this project, two datasets were used: 
1. DeepFashion
2. MVC

#### DeepFashion

[Link](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

DeepFashion is a 'large-scale clothes' database. It consists of **800,000** fashion images with four main categories: 

1. Category and Attribute Prediction
2. Consumer to Shop
3. Fashion Landmark
4. In-Shop Retrieval

Each category had slightly different data structures depending on the use. For example, Category and Attribute Prediction had labeled attributes whereas Fashion Landmark has coordinates to indicate specific 'landmarks' of a clothing (e.g. the collar, the sleeves, etc.). 

I used this dataset for part 1 and 2 of my project.

![][df-pic]

[df-pic]: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/attributes.jpg "DeepFashion Dataset (Category and Attribute Prediction)"

#### MVC
[Link](http://mvc-datasets.github.io/MVC/)

I was fortunate enough to gain access to this dataset in time to play with it for third part of my project. The dataset was consisted of **161,260** annotated images with relatively high resolutions as well as 244 attribute labels.

![][mvc-pic]

[mvc-pic]: http://mvc-datasets.github.io/MVC/MultiviewClothingImage.jpg "MVC Dataset"

---

## <a id="workflow"> Workflow

As mentioned previously, the workflow can be divided into three parts. 

### <a id="complete"> Complete Workflow

Here is a diagram that depicts the entire workflow.

![][complete-workflow-pic]

[complete-workflow-pic]: https://image.ibb.co/db7tS5/edited.png "Complete Workflow Diagram"

The user would feed their 'Reference Outfits' and 'Closet Clothes' into the first neural network. The Reference Outfits is a fancy way of saying what kind of clothes are out there, what kind of styles would you like to emulate.
The Closet Clothes is a dataset of clothes that you have on hand. 

These would be tagged as upper and lower accordingly. Then, these should be fed into the second network. Because attributes are different between top and bottom and the weights may be different (e.g. denim comes more often for lower clothings, and chiffon is an attribute more common for blouses which is an upper clothing), there are two neural networks for the second part.

The attriburtes tagged from the Reference Outfits will populate a cross-matrix which is basically a frequency table of what combination of attributes (aka outfits) have occurred.

From there, the user can input a clothing, indicate whether it is a top or bottomm. The system will reference the cross-matrix and spit out what attributes the input clothing has, what attributes it recommends (based on the Reference Outfits), and what clothings (based on the Closet Clothes) most closely matches the recommended attributes.


### <a id="p1"> Part I : Object Detection


![image](https://image.ibb.co/fN1bfQ/edited_P1.png)

I used Faster R-CNN implemented in Keras with backend TensorFlow to detect the bbox and classify the object with  DeepFashion's notation of "1" being upper clothes and "2" being lower clothes.


#### Data Preparation

`DeepFashion/DeepFashion/Part 1 (A) Extracting Data.ipynb`

This notebook will extract all the information from the four categories of the DeepFashion dataset.
Note that each category has its own way of parsing the data due to the nature of the data structure. 

Each file outputs like the following: `img_name x1 y1 x2 y2 upper_lower`

The `x1 y1 x2 y1` refers to the coordinate of the bbox of the item.
`upper_lower` can either be a 1 or 2 depending on whether the clothing was upper or lower.

`DeepFashion/DeepFashion/Part 1 (B) Splitting Data.ipynb`

This notebook goes about how we split the data between training and testing set.
Note: For this portion of this project we only used the category and landmark subsection.
The data is stored in `DeepFashion/DeepFashion/Data`.

#### Running Faster R-CNN

Faster R-CNN was selected to use because I wanted to be able to detect objects 'in the wild'. I wanted the system to recognize clothing objects regardless of pose and background noise. 

Examples of such taggings is shown below:

![p1](https://image.ibb.co/kbSK2F/Kakao_Talk_Photo_2017_07_27_09_10_42.jpg "object dtection in the streets") ![p2](https://image.ibb.co/cmYsNF/Kakao_Talk_Photo_2017_07_27_09_11_03.jpg "multiple detection") ![p3](https://image.ibb.co/fGQTwa/Kakao_Talk_Photo_2017_07_27_09_11_14.jpg "invariant of pose")

____#todo insert how to run Faster R-CNN

### <a id="p2"> Part II : Attribute Tagging

![image](https://image.ibb.co/g6vrEk/edited_P2.png)

This is the part that I struggled with due to the lack of clean/balanced data.


#### Data Preparation

`DeepFashion/DeepFashion/Part 2 (A) Cleaning Data for Attribute Tagging`

For this part, I used the 'Category and Attribute Prediction' portion of DeepFashion.
There were over 1000 attributes of 5 types (texture, fabric, shape, part, style).
The dataset consisted of 289,222 total images of which I only used 198,672 images (only used "1" and "2" for upper and lower clothes-- did not use "3" which were full-body clothes).

![image](https://image.ibb.co/dcF8Uv/image.png)

Originally, there were 1000 columns. I looked into the dataset and looked at the most frequent attributes per type and started whittling from there.
I noticed that several columns were relatively redundant, so I collapsed columns.

For example, the columns, `['dot', 'dots', 'dotted', 'polka dot']` would all be consolidated as `dot`. 

The matrix at the end was fairly sparse so I subsected the dataset even further by extracting rows that contained at least "2" or "3" attributes.
This significantly reduced my dataset.

The final column attributes looked like the following. As mentioned before, upper and lower attributes are separate.

![image](https://image.ibb.co/jVhNpv/image.png)

#### Balancing Imbalanced Data

Now, when I first trained the dataset and ran an inference, I ran into some serious trouble where it was tagging everything as denim.
After further inspection into the training dataset, it was highly _highly_ imbalanced.

In order to balance this dataset, I used a very cheap method of oversampling the underrepresented minority. 

`DeepFashion/DeepFashion/Part 2 (B) Balancing Imbalanced Data`

The underlying idea is that that I would 'pad' the imbalanced dataset with one-hot rows of that certain attribute. 

Example dataset before balancing:
![image](https://image.ibb.co/h9aaRa/image.png "example dataset before balancing")
Note that there are 2,223 cases for denim for only 496 cases for dot.

Example dataset after balancing
![image](https://image.ibb.co/cpKfsF/image.png "example dataset after balancing")
As you can see, the dataset is a lot more balanced.
However, the total size of the dataset increased as well.
This is something to take note of if you were to pick a better dataset to train on.

#### Note about Attempting to Use MVC Dataset for Tagging

I also tried using the MVC dataset instead but got worse results. I'm thinking it might be due to lack of training time.

Also, it might be worth it to look into various other loss functions to deal with the sparse matrix.

#### Running CNN for Attribute Tagging

____#todo insert how to run Attribute Tagging

---


### <a id = 'p3'> Part 3 : Recommendation based on Frequency

![image](https://image.ibb.co/kZjJuk/edited_P3.png)

After performing attribute tagging for the reference and closet clothes, we can proceed to the last portion.
The reference clothes make up the cross-matrix, which is basically a frequency table.
The idea is that things that occur often together, are more recommendable.

The workflow is that if the user inputs an item, the system will 
  1. Label it as a "1" or "2" depending on whether it is upper or lower
  2. Tag the attribute
  3. Refer to the cross-matrix to see what counter-attributes occur frequently
  4. Search and retrieve results from the closet clothes that most fulfill the counter-attributes.
  
#### POC of Recommender System

Note: The validity of the recommender system is highly dependent on the quality of the previous part. Because, I was not able to get part 2 to an adequate standard, I used a dummy matrix to display a proof of concept.

`DeepFashion/DeepFashion/Part 3 (A) POC Recommender with MVC`

I used the MVC dataset for this as I found their listing of attributes a lot cleaner than the DeepFashion dataset.

The purpose of this notebook is to show my overall thinking process of how a frequency table would be able to generate a list of a combination for recommendations.

![image](https://image.ibb.co/b1oJwa/image.png)

This is a sample dummy matrix that I worked with. (The numbers can be anything.)

The following shows an example of inputting a random item that is a lower clothing (labeled as "2").
It shows that the attributes for the pants are 'Cotton', 'Denim', Gray', 'HighRise', 'Polyester', 'StraightLeg'.
According to my dummy matrix, it recommends the upper clothing's attribute to be 'Athletic', 'Chambray', 'Wool', 'Zipper', 'Pink', 'TattooPrint', 'Yellow', 'Orange', 'Burgandy', 'Tropical'.

Given these attributes, it searched my closet clothes and retrieves the clothings that most matched the attributes.

![image](https://image.ibb.co/mTyOUv/image.png)


The parameter 'show = True' merely prints out all the relevant images so the user can visualize it. 

---

### <a id = 'future'> Future Direction and Ideas

Definitely a lot of things that can be improved. I divided up future ideas into two categories.

#### Higher Accuracy
This is just a list of technical changes that can be made for a higher accuracy of the workflow.

1. Cleaner dataset for Attribute Tagging
    * Separate clothings into Men's and Women's
    * Differentiate 'styles' and 'trends' (e.g. classy vs. casual, etc.)
    
2. Rebuild network architecture
    * Accomodate mutually-exclusive qualities
        * Sleeveless, short-sleeves, long-sleeves
    * Train on separate datasets (as opposed to one gigantic dataset) by qwuality to address data imbalance issue
3. Experiment with noramlization for frequency counts in the cross-matrix
    * Not all counts are the same. Perhaps run a logarthimic calculation?
    * Pad '0's with a count
    * When retrieving clothes the top matched attribrutes, make it scale with accordance to the frequency (attributes that have a higher frequency co-occurence are given higher priority)

#### Business Ideas
I think that the workflow is expandable and applicable to other fields.

1. Different pairings
    * What shoes to wear given a dress
    * What pocket square to wear given a tie
    * Can even extend to other 'pairing' questions such as, what painting to hang in the living room given the rest of the furniture
2. Item additions
    * 3+ item ensembles can be possible too
    * Perhaps: top, bottom, handbag
3. Incorporate user's preference
    * Add parameters and filtering abilities to match user's preferences
    * Users can manually adjust weights to certain attributes -- create a feedback/rating system
4. Scalability
    * Users can upload different datasets of 'closets' (closet 1, closet 2, closet 3)-- where each closet might be a different store's database
    * Users can upload different datasets of 'references' (style 1, style2, style 3)-- to copy a certain celebrity's trend or capture regional clothing trends
---

### <a id = 'thoughts'> Final Thoughts

It's interesting to note how the scope and direction of my project evolved throughout the span of one month. 

I definitely learned a lot through trial and error. A month flew by and I am glad to have this opportunity to make some sort of progress tackling a very 'real' business problem.

Long ways to go if this were to ever be in production but I believe it's a solid start and gets you thinking how combining the strengths/uses of various neural networks can result in a powerful workflow to solve everyday things.

#### Links to Previous Presentations
Initial Presentation: 
Mid Presentation:
Final Presentation:

---

### <a id = 'sources'> Sources and Acknowledgements

1. DeepFashion Dataset
2. MVC Dataset
3. Faster R-CNN (used for this project)
4. Inception v3 CNN (used for this project)
5. Acknowledgements
    * I would like to specially thank my mentor for his guidance throughout the project.
    * I would 

---




