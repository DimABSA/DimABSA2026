### File Naming Guidelines
When submitting your predictions on the Codabench task page:
 - Decide the target language(s) and domain(s). Each submission file corresponds to one language-domain combination.
 - For each language-domain combination, name the file `pred_[lang_code]_[domain].jsonl`, where <br>
		- `[lang_code]` represents a 3-letter [language code](#list-of-language-codes), and <br> 
		- `[domain]` represents a [domain](#list-of-domains).  
For example, Hausa predictions for the stance domain should be named `pred_hau_stance.jsonl`.
- If submitting for multiple languages or domains, submit one prediction file per language-domain combination. For example, submitting for multiple languages or domains would look like this:

```
subtask_1
├── pred_eng_restaurant.jsonl
├── pred_eng_laptop.jsonl
├── pred_eng_stance.jsonl
├── pred_deu_stance.jsonl
├── pred_hau_stance.jsonl
├── pred_jpn_hotel.jsonl
├── pred_jpn_finance.jsonl
├── pred_zho_restaurant.jsonl
├── pred_rus_restaurant.jsonl
├── pred_tat_restaurant.jsonl
└── pred_ukr_restaurant.jsonl
```

### Uploading for a Single Prediction File
If submitting a prediction file for a single language-domain combination (e.g., Hausa stance):
- Create a folder named `subtask_1` and place the prediction file (`pred_hau_stance.jsonl`) inside the folder.
- Zip the `subtask_1` folder.
- Navigate to the `My Submissions` section on Codabench, and upload the zipped folder.

That is, even a single prediction file must be placed inside a `subtask_1` folder before zipping and uploading.

### Uploading for Multiple Prediction Files
If submitting prediction files for multiple language-domain combinations (e.g., Hausa stance, English laptop, Chinese restaurant):
- Create a folder named `subtask_1` and place the prediction files (`pred_hau_stance.jsonl`, `pred_eng_laptop.jsonl`, `pred_zho_restaurant.jsonl`) inside the folder.
- Zip the `subtask_1` folder.
- Navigate to the - Navigate to the `My Submissions` section on Codabench, and upload the zipped folder.  
 section on Codabench, and upload the zipped folder.


## Important Instructions for Submitting Predictions on Codabench

When making submissions on Codabench, follow these guidelines to ensure **all your scores** remain visible on the leaderboard:

- **New Submissions Overwrite Previous Scores**
   - Every new submission will overwrite all your previous scores on the leaderboard.
   - If your submission omits a prediction file for any previously submitted language-domain combination, the score for that combination will be overwritten and marked `n/a` on the leaderboard.  
	 **Example**: If you submit a prediction file for Hausa stance, it will appear on the leaderboard. However, if your next submission includes prediction files for other language-domain combinations but omits a prediction file for Hausa stance, your previous Hausa stance score will be replaced with `n/a`.
- **Include All Predictions in Every Submission**
   - To ensure all your scores remain on the leaderboard, always include one prediction file for **every language-domain combination** you want to appear on the leaderboard in **each submission**, even for previously submitted combinations.


> Note: Before submitting, double-check that your submission folder includes prediction files for **all language-domain combinations** you want scored.


## Additional Resources

### List of Language Codes
- `deu`: German
- `eng`: English
- `hau`: Hausa
- `ibo`: Igbo
- `jpn`: Japanese
- `kin`: Kinyarwanda
- `ptb`: Brazilian Portuguese
- `ptm`: Mozambican Portuguese
- `rus`: Russian
- `swa`: Swahili
- `tat`: Tatar
- `twi`: Twi
- `ukr`: Ukrainian
- `vmw`: Emakhuwa
- `xho`: isiXhosa
- `zho`: Chinese

### List of Domains
- `restaurant`
- `laptop`
- `hotel`
- `movie`
- `stance`
- `finance`
