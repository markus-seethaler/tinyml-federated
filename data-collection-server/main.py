import serial
import pandas as pd
import os
import time
from datetime import datetime


def collect_sample(ser):
    data = []
    recording = False

    while True:
        line = ser.readline().decode('utf-8').strip()

        if line == "START":
            recording = True
            print("Recording started...")
            continue

        if line == "END":
            print("Recording complete!")
            break

        if recording:
            try:
                x, y, z = map(float, line.split(','))
                data.append([x, y, z])
            except ValueError:
                print(f"Skipped invalid line: {line}")

    return data


def save_recording(data, sample_id):
    # Create data directory if it doesn't exist
    if not os.path.exists('motion_data'):
        os.makedirs('motion_data')

    # Save the raw accelerometer data in a simple CSV format
    # Each row: timestamp_ms, acc_x, acc_y, acc_z
    df = pd.DataFrame(data, columns=['acc_x', 'acc_y', 'acc_z'])
    # Add timestamp column (10ms intervals)
    df.insert(0, 'timestamp_ms', range(0, len(df) * 10, 10))

    filename = f'motion_data/recording_{sample_id:04d}.csv'
    df.to_csv(filename, index=False)
    return filename


def delete_records(metadata_df, sample_ids):
    """
    Safely delete multiple records and their associated files, maintaining index integrity.

    Args:
        metadata_df: DataFrame containing the metadata
        sample_ids: List of sample IDs to delete

    Returns:
        Updated metadata DataFrame
    """
    try:
        # Check if any sample_ids exist
        invalid_ids = [sid for sid in sample_ids if sid not in metadata_df['sample_id'].values]
        if invalid_ids:
            print(f"Warning: Sample IDs {invalid_ids} not found in metadata.")
            sample_ids = [sid for sid in sample_ids if sid not in invalid_ids]
            if not sample_ids:
                return metadata_df

        # Delete files for all selected samples
        for sample_id in sample_ids:
            file_to_delete = metadata_df.loc[metadata_df['sample_id'] == sample_id, 'filename'].iloc[0]
            file_path = os.path.join('motion_data', file_to_delete)

            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"Warning: Data file {file_path} not found.")

        # Remove records from metadata
        metadata_df = metadata_df[~metadata_df['sample_id'].isin(sample_ids)].copy()

        # Reindex remaining samples
        metadata_df = metadata_df.reset_index(drop=True)

        # Update sample_ids to maintain sequential ordering
        metadata_df['sample_id'] = range(len(metadata_df))

        # Rename existing files to match new sample_ids
        for idx, row in metadata_df.iterrows():
            old_filename = row['filename']
            new_filename = f'recording_{idx:04d}.csv'

            if old_filename != new_filename:
                old_path = os.path.join('motion_data', old_filename)
                new_path = os.path.join('motion_data', new_filename)

                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    metadata_df.at[idx, 'filename'] = new_filename

        # Save updated metadata
        metadata_df.to_csv('motion_metadata.csv', index=False)
        print(f"Successfully deleted {len(sample_ids)} samples and updated indices.")

        return metadata_df

    except Exception as e:
        print(f"Error during deletion: {str(e)}")
        return metadata_df


def display_class_distribution(metadata_df):
    """Display the current distribution of classes in the dataset."""
    if len(metadata_df) == 0:
        print("\nNo samples collected yet.")
        return

    # Calculate class distribution
    class_counts = metadata_df['label'].value_counts().sort_index()
    total_samples = len(metadata_df)

    # Define class names
    class_names = {
        0: "No theft",
        1: "Carrying away",
        2: "Lock breaking"
    }

    print("\nCurrent Class Distribution:")
    print("-" * 50)
    print(f"Total samples: {total_samples}")
    print("-" * 50)

    # Display counts and percentages for each class
    for label in sorted(class_names.keys()):
        count = class_counts.get(label, 0)
        percentage = (count / total_samples) * 100
        print(f"{class_names[label]}: {count} samples ({percentage:.1f}%)")
    print("-" * 50)


def get_user_command():
    """Get and validate user command."""
    print("\nCommands:")
    print("1: Record new sample")
    print("2: Delete sample")
    print("3: Display class distribution")
    print("4: Exit")

    while True:
        cmd = input("\nEnter command (1-4): ")
        if cmd in ['1', '2', '3', '4']:
            return cmd
        print("Invalid command! Please enter 1, 2, 3, or 4")


def main():
    # Configure serial connection
    PORT = "/dev/ttyACM0"  # Change this to match your Arduino port
    BAUD_RATE = 115200

    # Create or load metadata index
    try:
        metadata_df = pd.read_csv('motion_metadata.csv')
        print("Loaded existing metadata index")
        # Display initial class distribution
        display_class_distribution(metadata_df)
    except FileNotFoundError:
        metadata_df = pd.DataFrame(columns=[
            'sample_id',
            'timestamp',
            'label',
            'filename'
        ])
        print("Created new metadata index")

    try:
        with serial.Serial(PORT, BAUD_RATE, timeout=1) as ser:
            print("Connected to Arduino")
            print("Press Ctrl+C to stop collecting data")

            while True:
                cmd = get_user_command()

                if cmd == '4':  # Exit
                    print("Exiting...")
                    break

                elif cmd == '3':  # Display distribution
                    display_class_distribution(metadata_df)
                    continue

                elif cmd == '2':  # Delete sample
                    if len(metadata_df) == 0:
                        print("No samples to delete!")
                        continue

                    print("\nCurrent samples:")
                    for _, row in metadata_df.iterrows():
                        print(f"ID: {row['sample_id']}, Label: {row['label']}, Time: {row['timestamp']}")

                    while True:
                        try:
                            sample_input = input(
                                "\nEnter sample ID or range to delete (e.g., '5' or '1-20', or -1 to cancel): ")
                            if sample_input == '-1':
                                break

                            # Check if input is a range
                            if '-' in sample_input:
                                try:
                                    start, end = map(int, sample_input.split('-'))
                                    if start > end:
                                        start, end = end, start
                                    sample_ids = list(range(start, end + 1))
                                except ValueError:
                                    print("Invalid range format! Use format 'start-end' (e.g., '1-20')")
                                    continue
                            else:
                                try:
                                    sample_ids = [int(sample_input)]
                                except ValueError:
                                    print("Invalid input! Enter a number or range (e.g., '5' or '1-20')")
                                    continue

                            # Validate range
                            max_id = metadata_df['sample_id'].max()
                            if any(sid < 0 or sid > max_id for sid in sample_ids):
                                print(f"Sample IDs must be between 0 and {max_id}")
                                continue

                            # Confirm deletion for large ranges
                            if len(sample_ids) > 5:
                                confirm = input(f"Are you sure you want to delete {len(sample_ids)} samples? (y/n): ")
                                if confirm.lower() != 'y':
                                    continue

                            metadata_df = delete_records(metadata_df, sample_ids)
                            break

                        except ValueError:
                            print("Please enter valid numbers!")
                    continue

                # cmd == '1': Record new sample
                print("\nPress Enter to start recording...")
                print("Sending record command to Arduino...")
                ser.write(b'r')

                # Collect data
                data = collect_sample(ser)

                if len(data) == 256:  # Verify we got the expected number of samples
                    # Get label from user
                    while True:
                        label = input("Enter label (0: No theft, 1: Carrying away, 2: Lock breaking): ")
                        if label in ['0', '1', '2']:
                            break
                        print("Invalid label! Please enter 0, 1, or 2")

                    # Generate sample ID
                    sample_id = len(metadata_df)

                    # Save recording to individual file
                    filename = save_recording(data, sample_id)

                    # Add metadata
                    new_row = {
                        'sample_id': sample_id,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'label': int(label),
                        'filename': os.path.basename(filename)
                    }

                    # Append to metadata dataframe
                    metadata_df = pd.concat([metadata_df, pd.DataFrame([new_row])], ignore_index=True)

                    # Save metadata
                    metadata_df.to_csv('motion_metadata.csv', index=False)
                    print(f"Saved recording {sample_id} with label {label}")

                    # Display updated class distribution
                    display_class_distribution(metadata_df)
                else:
                    print(f"Error: Received {len(data)} samples instead of 256")

    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    except serial.SerialException as e:
        print(f"Error with serial connection: {e}")
    finally:
        print("\nData collection complete")
        print(f"Total samples collected: {len(metadata_df)}")


if __name__ == "__main__":
    main()