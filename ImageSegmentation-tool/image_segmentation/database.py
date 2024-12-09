import sqlite3
from datetime import datetime

class ExperimentDatabase:
    def __init__(self, db_path="experiments.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            method TEXT NOT NULL,
            original_image_path TEXT NOT NULL,
            mask_path TEXT,
            segmented_image_path TEXT,
            accuracy REAL,
            iou REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            dice REAL,
            specificity REAL
        )
        """)
        self.conn.commit()


    def save_experiment(self, method, original_image_path, mask_path, segmented_image_path, metrics, segmentation_time):
        required_keys = ['accuracy', 'iou', 'precision', 'recall', 'f1', 'dice', 'specificity']
        metrics = {key: metrics.get(key, None) for key in required_keys}


        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO experiments (timestamp, method, original_image_path, mask_path, segmented_image_path, 
                                 accuracy, iou, precision, recall, f1, dice, specificity, segmentation_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            method,
            original_image_path,
            mask_path,
            segmented_image_path,
            metrics['accuracy'],
            metrics['iou'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['dice'],
            metrics['specificity'],
            segmentation_time
        ))
        self.conn.commit()
        print(f"Experiment saved with ID: {cursor.lastrowid}")

    def fetch_all_experiments(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT id, timestamp, method, original_image_path, mask_path, segmented_image_path,
               accuracy, iou, precision, recall, f1, dice, specificity, segmentation_time
        FROM experiments
        ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        experiments = []
        for row in rows:
            experiments.append({
                "id": row[0],
                "timestamp": row[1],
                "Method": row[2],
                "original_image_path": row[3],
                "mask_path": row[4],
                "segmented_image_path": row[5],
                "Accuracy": row[6],
                "IoU": row[7],
                "Precision": row[8],
                "Recall": row[9],
                "F1": row[10],
                "Dice": row[11],
                "Specificity": row[12],
                "SegmentationTime": row[13]
            })

        print(f"Fetched experiments: {experiments}")

        return experiments

    def close(self):
        self.conn.close()

