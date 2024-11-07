import random
import pandas as pd

# Sample augmentations

def random_delete(drug_name, num = 1):
    """Randomly delete a character."""
    if len(drug_name) <= 1:
        return drug_name
    idx = random.randint(0, len(drug_name) - 1)
    return drug_name[:idx] + drug_name[idx+num:]

def random_substitution(drug_name):
    """Introduce random noise by substituting a character."""
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    idx = random.randint(0, len(drug_name) - 1)
    return drug_name[:idx] + random.choice(alphabet) + drug_name[idx+1:]

def random_insertion(drug_name):
    """Insert a random character."""
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    idx = random.randint(0, len(drug_name))
    return drug_name[:idx] + random.choice(alphabet) + drug_name[idx:]

def augment_drug_name(drug_name):
    """Apply a random augmentation."""
    augmentations = [random_delete, random_insertion]
    augmentation_func = random.choices(augmentations, weights=[0.9, 0.1], k = 1)[0]# Randomly choose an augmentation
    # augmentation_func = random_delete

    if augmentation_func == random_delete:
        # if len(drug_name)<8:
        #     dele_num = 1
        # else:
        #     dele_num = max(random.randint(1, len(drug_name)//2)//2, 1)
        potention_num = [1,2,3]
        dele_num = random.choices(potention_num, weights=[0.6, 0.3, 0.1], k = 1)[0]
        return augmentation_func(drug_name, dele_num)
    else:
        return augmentation_func(drug_name)

# Function to apply augmentation to the entire dataset
def apply_augmentation_to_dataset(df, column_name, augment_ratio=0.6, ep = 5):
    """
    Apply augmentations to a portion of the dataset.

    Args:
    df: DataFrame containing the dataset.
    column_name: The column containing drug names to augment.
    augment_ratio: The fraction of the dataset to augment. Default is 0.5 (50% of the dataset).

    Returns:
    A new DataFrame with original and augmented data combined.
    """
    for i in range(ep):
        augmented_data = []

        # Iterate over the dataset
        for _, row in df.iterrows():
            original_name = row[column_name]
            # Store the original row
            augmented_data.append(row)

            # Apply augmentation based on the specified ratio
            if len(original_name) >= 16 and random.random() < augment_ratio:
                augmented_row = row.copy()
                if len(augmented_row[column_name].split(' ')) > 1:
                    first_part = original_name.split(' ')[0]
                    unchanged = ' '.join(original_name.split(' ')[1:])
                    augmented_row[column_name] = augment_drug_name(first_part) + ' ' + unchanged
                else:
                    augmented_row[column_name] = augment_drug_name(original_name)
                augmented_data.append(augmented_row)

        # Return a new DataFrame with both original and augmented data
        augmented_df = pd.DataFrame(augmented_data).reset_index(drop = True)
        df = augmented_df
    return augmented_df

if __name__ == '__main__':

    data = {
        'DRUG_NAME': ['ASPIRIN', 'IBUPROFEN', 'PARACETAMOL', 'AMOXICILLIN', 'METFORMIN', 'METFORMIN AF DF', 'METFORMIN DF'],
        'ART': [1, 2, 3, 4, 5, 6, 7],  # Example target labels
        'TYPE': ['OTC', 'OTC', 'OTC', 'RX', 'RX', 'RX', 'RX']
    }
    df = pd.DataFrame(data)
    # Example: Load your dataset (you can replace this with your actual dataset)}
    print(df)

    # Apply augmentation to the dataset
    augmented_df = apply_augmentation_to_dataset(df, column_name='DRUG_NAME', augment_ratio=0.5)

    print(augmented_df)

