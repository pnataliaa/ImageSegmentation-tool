from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.dropdown import DropDown
from kivy.graphics import RoundedRectangle
from kivy.core.window import Window
from kivy.clock import Clock
from tensorflow.keras.models import load_model
from functools import partial
from fuzzycmeans import segment_image_with_fuzzy_cmeans
from tresholding import apply_modified_otsu_with_multithreshold
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score, f1_score
from kivy.uix.image import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from kivy.uix.boxlayout import BoxLayout
import datetime as dt
from kivy.graphics.texture import Texture
from kivy.core.image import Image as CoreImage
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.textinput import TextInput
import time
from kivy.uix.checkbox import CheckBox
import numpy as np
from kivy.uix.widget import Widget
from reportlab.platypus import Image as ReportLabImage, Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from kivy.uix.scrollview import ScrollView
from database import ExperimentDatabase
from kivy.uix.gridlayout import GridLayout
from kivy.core.text import LabelBase
from kivy.graphics import Color, Line
from kivy.uix.label import Label
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import csv
import io
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.datatables import MDDataTable
from kivy.metrics import dp
from kivymd.app import MDApp
import os
from pathlib import Path
import signal

signal.signal(signal.SIGTRAP, signal.SIG_IGN)

CLASS_COLORS = {
    0: [60, 16, 152],
    1: [132, 41, 246],
    2: [110, 193, 228],
    3: [254, 221, 58],
    4: [226, 169, 41],
    5: [155, 155, 155],
}

def decode_segmentation(mask, class_colors):
    decoded_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        decoded_mask[mask == class_idx] = color
    return decoded_mask

class StartScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app

        with self.canvas.before:
            Color(0.2, 0.2, 0.2, 1)
            self.bg_rect = RoundedRectangle(size=self.size, pos=self.pos)
        self.bind(size=self.update_bg_rect, pos=self.update_bg_rect)


        frame = BoxLayout(
            orientation='vertical',
            padding=[80, 220, 80, 50],
            spacing=20,
            size_hint=(None, None),
            size=(1800, 1400),
            pos_hint={'center_x': 0.5, 'center_y': 0.45},
        )

        with frame.canvas.before:
            Color(0.95, 0.95, 0.95, 1)
            self.rect = RoundedRectangle(size=frame.size, pos=frame.pos, radius=[20])
        frame.bind(size=self.update_rect, pos=self.update_rect)

        LabelBase.register(name="SpecialElite", fn_regular="SpecialElite-Regular.ttf")

        frame.add_widget(BoxLayout(size_hint=(1, None), height=50))
        title = Label(
            text="Narzędzie do eksperymentalnej segmentacji obrazów",
            font_size=65,
            bold=True,
            font_name="SpecialElite",
            halign="center",
            valign="middle",
            size_hint=(1, None),
            height=120,
            color=(0.2, 0.4, 0.8, 1),
        )
        title.bind(size=title.setter('text_size'))
        frame.add_widget(title)

        file_layout = BoxLayout(orientation='vertical', spacing=20, size_hint=(1, None), height=250)
        self.file_chooser_button = Button(
            text="Wybierz plik",
            font_size=35,
            bold=True,
            size_hint=(1, None),
            height=90,
            background_color=(0.5, 0.5, 0.5, 1),
            color=(1, 1, 1, 1),
        )
        self.file_chooser_button.bind(on_press=self.show_file_chooser)
        file_layout.add_widget(self.file_chooser_button)

        self.file_label = Label(
            text="Nie wybrano pliku.",
            font_size=28,
            bold=True,
            color=(0, 0, 0, 1),
            halign="center",
            valign="middle",
            size_hint=(1, None),
            height=50,
        )
        file_layout.add_widget(self.file_label)
        frame.add_widget(file_layout)

        method_layout = BoxLayout(orientation='vertical', spacing=25, size_hint=(1, None), height=200)
        self.method_dropdown = DropDown()
        for method in ["Otsu", "U-Net", "C-means"]:
            btn = Button(
                text=method,
                size_hint_y=None,
                height=70,
                font_size=30,
                background_color=(0.5, 0.5, 0.5, 1),
                color=(1, 1, 1, 1),
            )
            btn.bind(on_release=lambda btn: self.method_dropdown.select(btn.text))
            self.method_dropdown.add_widget(btn)

        self.method_button = Button(
            text="Wybierz metodę segmentacji",
            font_size=35,
            size_hint=(1, None),
            height=90,
            bold=True,
            background_color=(0.5, 0.5, 0.5, 1),
            color=(1, 1, 1, 1),
        )
        self.method_button.bind(on_release=self.method_dropdown.open)
        self.method_dropdown.bind(on_select=lambda instance, x: setattr(self.method_button, 'text', x))
        method_layout.add_widget(self.method_button)
        frame.add_widget(method_layout)

        metrics_label = Label(
            text="Wybierz metryki:",
            font_size=35,
            bold=True,
            color=(0, 0, 0, 1),
            halign="left",
            valign="middle",
            size_hint=(1, None),
            height=60,
        )
        frame.add_widget(metrics_label)

        metrics_layout = BoxLayout(orientation='vertical', spacing=30, size_hint=(1, None))
        metrics_layout.bind(minimum_height=metrics_layout.setter('height'))
        scrollable_metrics = ScrollView(size_hint=(1, None), height=450)
        scrollable_metrics.add_widget(metrics_layout)

        self.metric_checkboxes = {}
        for metric in self.app.selected_metrics:
            checkbox_layout = BoxLayout(orientation='horizontal', spacing=15, size_hint=(1, None), height=60)
            checkbox_label = Label(
                text=metric,
                font_size=30,
                bold=True,
                color=(0, 0, 0, 1),
                size_hint=(0.8, 1),
            )
            checkbox = CheckBox(active=self.app.selected_metrics[metric], size_hint=(0.2, 1))
            checkbox.bind(active=partial(self.set_metric_selection, metric))
            checkbox_layout.add_widget(checkbox_label)
            checkbox_layout.add_widget(checkbox)
            metrics_layout.add_widget(checkbox_layout)

        frame.add_widget(scrollable_metrics)

        start_button = Button(
            text="Gotowe",
            font_size=50,
            bold=True,
            size_hint=(1, None),
            height=120,
            background_color=(0.2, 0.3, 0.6, 1),
            color=(1, 1, 1, 1),
        )
        start_button.bind(on_press=self.start_main_screen)
        frame.add_widget(start_button)

        self.add_widget(frame)

    def set_metric_selection(self, metric, instance, is_selected):
        self.app.selected_metrics[metric] = is_selected

    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def update_bg_rect(self, instance, value):
        self.bg_rect.pos = instance.pos
        self.bg_rect.size = instance.size

    def show_file_chooser(self, instance):
        self.filechooser = FileChooserListView(filters=['*.png', '*.jpg', '*.jpeg'], size_hint=(1, 0.8))
        popup_content = BoxLayout(orientation='vertical')
        popup_content.add_widget(self.filechooser)

        select_button = Button(
            text="Wybierz",
            size_hint=(1, 0.2),
            font_size=30,
            bold=True,
            background_color=(0.4, 0.4, 0.4, 1),
            color=(1, 1, 1, 1),
        )
        select_button.bind(on_press=self.select_file)
        popup_content.add_widget(select_button)

        self.popup = Popup(
            title="Wybierz obraz",
            content=popup_content,
            size_hint=(0.9, 0.9),
        )
        self.popup.open()

    def select_file(self, instance):
        selection = self.filechooser.selection
        if selection:
            selected_path = selection[0]

            supported_formats = ['.png', '.jpg', '.jpeg']
            file_extension = os.path.splitext(selected_path)[1].lower()

            if file_extension not in supported_formats:
                popup = Popup(
                    title="Błąd",
                    content=Label(
                        text="Nieobsługiwany format pliku. Wybierz plik w formacie PNG, JPG lub JPEG.",
                        font_size=20
                    ),
                    size_hint=(0.8, 0.3)
                )
                popup.open()
                return

            self.app.selected_image_path = selected_path
            self.file_label.text = f"Wybrano plik: {os.path.basename(selected_path)}"
            self.app.selected_mask_path = self.app.find_mask_for_image(selected_path)


            if self.app.selected_mask_path:
                print(f"Znaleziono maskę: {self.app.selected_mask_path}")
            else:
                print("Nie znaleziono maski dla wybranego obrazu.")

            self.popup.dismiss()

    def start_main_screen(self, instance):
        if not hasattr(self.app, 'selected_image_path'):
            self.file_label.text = "Najpierw wybierz plik!"
            return
        if self.method_button.text == "Wybierz metodę segmentacji":
            self.file_label.text = "Najpierw wybierz metodę segmentacji!"
            return
        self.app.segment_button.text = self.method_button.text
        self.manager.current = "main"
        self.app.run_segmentation(None)


class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        self.database = ExperimentDatabase()
        self.unet_model = load_model("unet_model_satellite.keras")
        self.metrics_data = pd.read_csv("segmentation_results.csv")
        self.original_image_display = Image()
        self.segmented_image_display = Image()

    def build(self):

        Window.size = (1400, 900)

        self.report_data = {}
        self.last_metrics = None
        self.selected_metrics = {
            "Dokładność": True,
            "IOU": True,
            "Precyzja": True,
            "Czułość": True,
            "F1": True,
            "Dice": True,
            "Specyficzność": True
        }

        self.screen_manager = ScreenManager()

        self.start_screen = StartScreen(app=self, name="start")
        self.screen_manager.add_widget(self.start_screen)

        self.main_screen = Screen(name="main")
        self.main_screen.add_widget(self.build_main_layout())
        self.screen_manager.add_widget(self.main_screen)

        return self.screen_manager

    def save_experiment_to_db(self):
        if not hasattr(self, 'report_data'):
            print("No data to save.")
            return

        method = self.segment_button.text
        if method == "Wybierz metodę segmentacji":
            print("No segmentation method selected.")
            return

        selected_metrics = {metric: selected for metric, selected in self.selected_metrics.items() if selected}
        metric_results = {
            metric: self.last_metrics.get(key, "N/A")
            for metric, key in [
                ("Dokładność", "accuracy"),
                ("IOU", "iou"),
                ("Precyzja", "precision"),
                ("Czułość", "recall"),
                ("F1", "f1"),
                ("Dice", "dice"),
                ("Specyficzność", "specificity")
            ]
            if metric in selected_metrics
        }

        segmented_image_path = f"segmented_{method.lower()}.png"
        if hasattr(self, 'segmented_image') and self.segmented_image is not None:
            cv2.imwrite(segmented_image_path, self.segmented_image)

        self.database.save_experiment(
            method=method,
            selected_metrics=selected_metrics,
            metric_results=metric_results,
            original_image_path=self.selected_image_path,
            mask_path=getattr(self, 'selected_mask_path', None),
            segmented_image_path=segmented_image_path
        )

    def build_main_layout(self):
        main_layout = BoxLayout(orientation='horizontal', padding=10)
        left_layout = BoxLayout(orientation='vertical', spacing=10, size_hint=(0.65, 1))

        with main_layout.canvas.before:
            Color(0, 0, 0, 1)
            self.bg_rect = RoundedRectangle(size=main_layout.size, pos=main_layout.pos)
        main_layout.bind(size=self.update_rect, pos=self.update_rect)

        image_layout = BoxLayout(orientation='horizontal', spacing=20, size_hint=(1, 0.55), padding=[20, 0, 20, 0])
        original_image_box = self.create_image_box("Obraz Oryginalny", self.original_image_display, size_hint=(1, 1))
        segmented_image_box = self.create_image_box("Obraz po Segmentacji", self.segmented_image_display,
                                                    size_hint=(1, 1))
        image_layout.add_widget(original_image_box)
        image_layout.add_widget(segmented_image_box)
        left_layout.add_widget(image_layout)

        file_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, None), height=120,
                                padding=[20, 0, 20, 0])

        self.file_chooser_button = Button(
            text="Wybierz plik",
            font_size=35,
            bold=True,
            size_hint=(0.8, None),
            height=120,
            background_color=(0.2, 0.6, 0.8, 1),
            color=(1, 1, 1, 1),
        )
        self.file_chooser_button.bind(on_press=self.show_file_chooser)
        file_layout.add_widget(self.file_chooser_button)

        db_button = Button(
            size_hint=(None, None),
            size=(120, 120),
            background_normal='database.png',
            background_down='database.png',
            background_color=(1, 1, 1, 1),
            border=(0, 0, 0, 0),
        )
        db_button.bind(on_press=self.show_database_view)
        file_layout.add_widget(db_button)

        left_layout.add_widget(file_layout)

        right_layout = BoxLayout(orientation='vertical', spacing=10, padding=[20, 0, 20, 20], size_hint=(0.35, 1))

        self.dropdown = DropDown()
        for method in ["Otsu", "U-Net", "C-means"]:
            btn = Button(text=method, size_hint_y=None, height=44, font_size=30, bold=True)
            btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
            self.dropdown.add_widget(btn)

        self.segment_button = Button(text='Wybierz metodę segmentacji', size_hint=(1, 0.1), font_size=30, bold=True)
        self.segment_button.bind(on_release=self.open_method_selection_popup)
        self.dropdown.bind(on_select=lambda instance, x: setattr(self.segment_button, 'text', x))

        segmentation_button = Button(text='Uruchom segmentację', size_hint=(1, 0.1), font_size=30, bold=True)
        segmentation_button.bind(on_press=self.run_segmentation)

        download_report_button = Button(text='Pobierz raport', size_hint=(1, 0.1), font_size=30, bold=True)
        download_report_button.bind(on_press=self.save_report)

        details_button = Button(text='Szczegóły segmentacji', size_hint=(1, 0.1), font_size=30, bold=True)
        details_button.bind(on_release=self.show_segmentation_details)

        select_metrics_button = Button(text='Wybierz metryki', size_hint=(1, 0.1), font_size=30, bold=True)
        select_metrics_button.bind(on_release=self.open_metric_selection_popup)

        self.metrics_box = BoxLayout(orientation='vertical', padding=10, spacing=5, size_hint=(1, 0.5))
        with self.metrics_box.canvas.before:
            Color(0.2, 0.2, 0.2, 0.8)
            RoundedRectangle(pos=self.metrics_box.pos, size=self.metrics_box.size, radius=[20])
            self.metrics_box.bind(pos=self.update_rounded_rectangle, size=self.update_rounded_rectangle)

        title_label = Label(
            text="NAJWAŻNIEJSZE METRYKI",
            font_size=40,
            bold=True,
            color=(1, 1, 1, 1),
            size_hint_y=None,
            height=80,
        )
        self.metrics_box.add_widget(title_label)

        self.processing_time_label = self.add_card(self.metrics_box, "Czas przetwarzania: ", size=40, height=100)

        self.metrics_box.opacity = 0
        self.metrics_box.disabled = True

        self.metric_labels = {
            "Dokładność": self.add_card(self.metrics_box, "Dokładność:", size=40, height=100),
            "IOU": self.add_card(self.metrics_box, "IOU:", size=40, height=100),
            "Precyzja": self.add_card(self.metrics_box, "Precyzja:", size=40, height=100),
            "Czułość": self.add_card(self.metrics_box, "Czułość:", size=40, height=100),
            "F1": self.add_card(self.metrics_box, "F1:", size=40, height=100),
            "Dice": self.add_card(self.metrics_box, "Dice:", size=40, height=100),
            "Specyficzność": self.add_card(self.metrics_box, "Specyficzność:", size=40, height=100),

        }

        right_layout.add_widget(self.segment_button)
        right_layout.add_widget(segmentation_button)
        right_layout.add_widget(download_report_button)
        right_layout.add_widget(details_button)
        right_layout.add_widget(select_metrics_button)
        right_layout.add_widget(self.metrics_box)

        main_layout.add_widget(left_layout)
        main_layout.add_widget(right_layout)

        Clock.schedule_interval(self.update, 0.1)
        self.hover_popup_open = False

        return main_layout

    def create_image_box(self, title, image_widget, size_hint=(1, 1)):
        box = BoxLayout(orientation='vertical', padding=10, spacing=10, size_hint=size_hint)

        with box.canvas.before:
            Color(0.9, 0.9, 0.9, 1)
            self.bg_rect = RoundedRectangle(size=box.size, pos=box.pos, radius=[10])
        box.bind(size=self.update_rect, pos=self.update_rect)

        label = Label(
            text=title,
            font_size=48,
            bold=True,
            color=(0.1, 0.1, 0.1, 1),
            size_hint=(1, None),
            font_name="SpecialElite",
            height=85,
            halign="center",
            valign="middle"
        )
        label.bind(size=label.setter('text_size'))

        image_widget.allow_stretch = True
        image_widget.keep_ratio = True

        box.add_widget(label)
        box.add_widget(image_widget)
        return box

    def update_rect(self, instance, value):
        for instruction in instance.canvas.before.children:
            if isinstance(instruction, RoundedRectangle):
                instruction.pos = instance.pos
                instruction.size = instance.size

    def open_method_selection_popup(self, instance):
        content = BoxLayout(orientation='vertical', spacing=20)
        method_group = ["Otsu", "U-Net", "C-means"]

        for method in method_group:
            toggle_button = ToggleButton(
                text=method,
                group="segmentation_method",
                size_hint=(1, None),
                height=100
            )
            toggle_button.bind(on_press=partial(self.set_method_selection, method))
            content.add_widget(toggle_button)

        close_button = Button(
            text="Zamknij",
            size_hint=(1, None),
            height=80,
            background_color=(0.1, 0.3, 0.7, 1)
        )
        close_button.bind(on_release=lambda x: self.method_selection_popup.dismiss())
        content.add_widget(close_button)

        self.method_selection_popup = Popup(
            title="Wybierz metodę segmentacji",
            content=content,
            size_hint=(0.4, 0.4),
            auto_dismiss=False
        )
        self.method_selection_popup.open()

    def set_method_selection(self, method, instance):
        self.segment_button.text = method
        print(f"Wybrana metoda: {method}")

    def open_metric_selection_popup(self, instance):
        num_metrics = len(self.selected_metrics)
        popup_height = 0.15 + num_metrics * 0.1
        popup_height = min(popup_height, 0.6)
        popup_width = 0.45

        content = BoxLayout(orientation='vertical', padding=30, spacing=25)

        for metric in self.selected_metrics:
            metric_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=70, padding=(10, 0))

            label = Label(
                text=f"[b]{metric}[/b]",
                font_size=40,
                markup=True,
                halign="left",
                valign="middle",
                size_hint_x=0.8
            )
            label.bind(size=label.setter('text_size'))

            checkbox = CheckBox(active=self.selected_metrics[metric], size_hint_x=0.2)
            checkbox.bind(active=lambda cb, value, m=metric: self.set_metric_selection(m, value))

            metric_layout.add_widget(label)
            metric_layout.add_widget(checkbox)
            content.add_widget(metric_layout)

        close_button = Button(
            text="Zamknij",
            size_hint_y=None,
            height=60,
            font_size=30,
            bold=True,
            background_color=(0.1, 0.3, 0.7, 1),
            color=(1, 1, 1, 1),
        )
        close_button.bind(on_release=lambda x: self.metric_selection_popup.dismiss())
        content.add_widget(close_button)

        self.metric_selection_popup = Popup(
            title="Wybierz metryki",
            content=content,
            size_hint=(popup_width, popup_height),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        self.metric_selection_popup.open()

    def set_metric_selection(self, metric, is_selected):
        self.selected_metrics[metric] = is_selected
        self.update_metrics_display()
        if self.last_metrics is not None:
            self.display_metrics(self.last_metrics)

    def update_metrics_display(self):
        for metric, label in self.metric_labels.items():
            label.text = f"{metric}:" if self.selected_metrics.get(metric, False) else ""

    def add_card(self, parent, text, size=35, height=60, padding=10):
        box = BoxLayout(orientation='horizontal', padding=padding, size_hint_y=None, height=height)
        with box.canvas.before:
            Color(0.9, 0.9, 0.9, 1)
            RoundedRectangle(pos=box.pos, size=box.size, radius=[15])
        box.bind(pos=self.update_rounded_rectangle, size=self.update_rounded_rectangle)

        label = Label(
            text=f"[b]{text}[/b]",
            font_size=size,
            markup=True,
            halign="center",
            valign="middle",
            color=(0, 0, 0, 1)
        )
        label.bind(size=label.setter('text_size'))
        box.add_widget(label)

        parent.add_widget(box)
        return label

    def update_rounded_rectangle(self, instance, value):
        for instruction in instance.canvas.before.children:
            if isinstance(instruction, RoundedRectangle):
                instruction.pos = instance.pos
                instruction.size = instance.size

    def show_file_chooser(self, instance):
        self.filechooser = FileChooserListView(filters=['*.png', '*.jpg', '*.jpeg'], size_hint=(1, 0.8))

        choose_button = Button(text="Wybierz", size_hint=(1, 0.2), background_color=(0.3, 0.6, 0.8, 1),
                               color=(1, 1, 1, 1), font_size=30, bold=True)
        choose_button.bind(on_press=self.load_image)

        content = BoxLayout(orientation='vertical')
        content.add_widget(self.filechooser)
        content.add_widget(choose_button)

        self.popup = Popup(title="Wybierz obraz", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def find_mask_for_image(self, image_path):
        image_basename = os.path.basename(image_path).split('.')[0]

        jpeg_images_dir = os.path.dirname(image_path)

        mask_dir = None

        if "retina_blood_vessel" in jpeg_images_dir:
            dataset_dir = os.path.dirname(os.path.dirname(image_path))
            if "test" in image_path:
                mask_dir = os.path.join(dataset_dir, "mask")
            elif "train" in image_path:
                mask_dir = os.path.join(dataset_dir, "mask")
            else:
                print(f"Cannot determine mask directory for: {image_path}")
                return None

        elif "JPEGImages" in jpeg_images_dir:
            mask_dir = jpeg_images_dir.replace("JPEGImages", "SegmentationClass")
        elif "images" in jpeg_images_dir:
            mask_dir = jpeg_images_dir.replace("images", "masks")
        elif "Image" in jpeg_images_dir:
            mask_dir = jpeg_images_dir.replace("Image", "Mask")

        if not mask_dir:
            print(f"Unknown dataset structure for: {jpeg_images_dir}")
            return None

        mask_path = os.path.join(mask_dir, f"{image_basename}.png")

        print(f"Image Path: {image_path}")
        print(f"Expected Mask Path: {mask_path}")

        if os.path.exists(mask_path):
            print(f"Mask found: {mask_path}")
            return mask_path
        else:
            print(f"No corresponding mask found for image: {image_basename}")
            return None

    def load_image(self, instance):
        self.clear_previous_segmentations()
        self.reset_segmentation_settings()

        selection = self.filechooser.selection
        if selection:
            self.selected_image_path = selection[0]
            self.popup.dismiss()

            supported_formats = ['.png', '.jpg', '.jpeg']
            file_extension = os.path.splitext(self.selected_image_path)[1].lower()

            if file_extension not in supported_formats:
                popup = Popup(
                    title="Błąd",
                    content=Label(text="Nieobsługiwany format pliku. Wybierz plik w formacie PNG, JPG lub JPEG.",
                                  font_size=20),
                    size_hint=(0.8, 0.3)
                )
                popup.open()
                return

            img = cv2.imread(self.selected_image_path)
            if img is None:
                popup = Popup(
                    title="Błąd",
                    content=Label(text="Nie można załadować pliku. Sprawdź, czy plik jest poprawny.", font_size=20),
                    size_hint=(0.8, 0.3)
                )
                popup.open()
                return

            self.selected_mask_path = self.find_mask_for_image(self.selected_image_path)
            if not self.selected_mask_path:
                print("No corresponding mask found for this image. Metrics may be incomplete.")

            if self.selected_mask_path:
                print(f"Maska odnaleziona i ustawiona: {self.selected_mask_path}")
            else:
                print("Nie znaleziono maski dla wybranego obrazu.")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_image = img_rgb

            try:
                original_img_texture = self.array_to_texture(img_rgb)
                self.original_image_display.texture = original_img_texture
                print("Image displayed successfully in the app.")
            except Exception as e:
                print(f"Error displaying image: {e}")

    def clear_previous_segmentations(self):
        self.original_image_display.texture = None
        self.segmented_image_display.texture = None

        for file_name in os.listdir():
            if file_name.endswith(".png") and "segmented" in file_name:
                os.remove(file_name)
                print(f"Usunięto plik segmentacji: {file_name}")

    def reset_segmentation_settings(self):
        self.segment_button.text = 'Wybierz metodę segmentacji'

        for metric in self.selected_metrics:
            self.selected_metrics[metric] = False

        self.update_metrics_display()

        self.metrics_box.opacity = 0
        self.metrics_box.disabled = True

        print("Resetowano ustawienia segmentacji i wyboru metryk.")

    def array_to_texture(self, img):
        try:
            if len(img.shape) == 3 and img.shape[2] == 3:
                colorfmt = 'rgb'
            else:
                colorfmt = 'luminance'

            img = np.flip(img, axis=0)

            texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt=colorfmt)
            texture.blit_buffer(img.tobytes(), colorfmt=colorfmt, bufferfmt='ubyte')
            return texture
        except Exception as e:
            print(f"Error in array_to_texture: {e}")
            raise

    def normalize_path(self, full_path):
        return "/".join(full_path.split("/")[-3:])

    def get_metrics_for_unet(self, image_path, metrics_df):
        try:
            normalized_path = self.normalize_path(image_path)
            filtered_df = metrics_df[metrics_df['image_path'] == normalized_path]

            if filtered_df.empty:
                print(f"No metrics found for file path: {normalized_path}")
                return None

            metrics_row = filtered_df.iloc[0]

            metrics = {
                "iou": metrics_row["iou"],
                "precision": metrics_row["precision"],
                "recall": metrics_row["recall"],
                "f1": metrics_row["f1"],
                "dice": metrics_row["dice"],
                "specificity": metrics_row["specificity"],
                "accuracy": metrics_row["accuracy"],
            }
            return metrics
        except KeyError as e:
            print(f"Error: The required key does not exist in DataFrame - {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def run_segmentation(self, instance):
        if hasattr(self, 'selected_mask_path') and self.selected_mask_path:
            true_mask = cv2.imread(self.selected_mask_path, cv2.IMREAD_GRAYSCALE)
            if true_mask is not None:
                print("Maska została poprawnie załadowana.")
                self.true_mask = true_mask
            else:
                print("Błąd podczas ładowania maski.")
        else:
            true_mask = None
            print("Nie znaleziono odpowiadającej maski referencyjnej.")


        if not hasattr(self, 'selected_image_path'):
            self.accuracy_label.text = "Najpierw wybierz obraz!"
            return

        method = self.segment_button.text
        if method == "Wybierz metodę segmentacji":
            self.accuracy_label.text = "Najpierw wybierz metodę segmentacji!"
            return

        img = cv2.imread(self.selected_image_path)
        if img is None:
            print("Błąd: Nie można wczytać obrazu.")
            return

        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            self.original_image_display.texture = self.array_to_texture(img)
        except Exception as e:
            print(f"Błąd podczas wyświetlania oryginalnego obrazu: {e}")

        segmented_image = None
        start_time = time.time()

        if method == "Otsu":
            segmented_image = apply_modified_otsu_with_multithreshold(img_gray)
            segmented_image = cv2.resize(segmented_image, (self.original_image.shape[1], self.original_image.shape[0]))
            self.segmented_image = segmented_image
            self.display_segmented_image(segmented_image)

            if true_mask is not None:
                if segmented_image.shape != true_mask.shape:
                    if len(segmented_image.shape) == 3 and segmented_image.shape[2] == 3:
                        print(
                            f"Reducing segmented_image dimensions from {segmented_image.shape} to match true_mask {true_mask.shape}")
                        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
                    elif segmented_image.shape[0:2] != true_mask.shape:
                        print(f"Resizing segmented_image from {segmented_image.shape} to {true_mask.shape}")
                        segmented_image = cv2.resize(segmented_image, (true_mask.shape[1], true_mask.shape[0]))

            if true_mask is not None:
                metrics = self.calculate_metrics(true_mask, segmented_image)
                self.report_data = {'otsu': metrics}
                self.last_metrics = metrics
                self.display_metrics(metrics)

        elif method == "U-Net":
            img_resized = cv2.resize(self.original_image, (256, 256))
            img_normalized = img_resized / 255.0
            img_normalized = np.expand_dims(img_normalized, axis=0)
            prediction = self.unet_model.predict(img_normalized)
            predicted_mask = np.argmax(prediction[0], axis=-1)

            segmented_image = decode_segmentation(predicted_mask, CLASS_COLORS)
            self.segmented_image = segmented_image
            self.display_segmented_image(segmented_image)


            if hasattr(self, 'metrics_data') and self.metrics_data is not None:
                print(f"Metrics data available: {self.metrics_data}")
                unet_metrics = self.get_metrics_for_unet(self.selected_image_path, self.metrics_data)
                if unet_metrics:
                    self.last_metrics = unet_metrics
                    self.display_metrics(unet_metrics)
                    self.report_data = {'u-net': unet_metrics}
                    print(f"Displayed U-Net metrics for {self.selected_image_path}: {unet_metrics}")
                else:
                    print(f"No metrics found for the selected image: {self.selected_image_path}")
            else:
                print("Metrics data is not available for U-Net.")

        elif method == "C-means":
            self.show_cluster_input_popup()
            return

        end_time = time.time()
        processing_time = end_time - start_time
        self.processing_time_label.text = f"Czas przetwarzania: {processing_time:.2f} sekundy"
        print(f"Czas przetwarzania: {processing_time:.2f} sekundy")

        Clock.schedule_once(
            lambda _: self.show_save_popup(
                method=self.segment_button.text,
                default_name=f"{len(self.database.fetch_all_experiments())}_{self.segment_button.text.lower()}_{dt.datetime.now().strftime('%Y-%m-%d')}.png"
            ),
            10
        )

        if true_mask is None:
            error_popup = Popup(
                title="Brak maski referencyjnej",
                content=Label(
                    text="Nie można obliczyć metryk. Nie znaleziono odpowiadającej maski referencyjnej.",
                    font_size=24,
                    halign="center",
                    valign="middle"
                ),
                size_hint=(0.4, 0.2),
            )
            error_popup.open()
            self.metrics_box.opacity = 0
            self.metrics_box.disabled = True
            return

        segmented_image_path = f"segmented_{method.lower()}.png"
        cv2.imwrite(segmented_image_path, segmented_image)
        self.segmented_image_path = segmented_image_path
        print(f"Segmentowany obraz zapisano jako: {segmented_image_path}")

        if segmented_image is not None:
            if segmented_image.shape != self.original_image.shape[:2]:
                segmented_image = cv2.resize(segmented_image,
                                             (self.original_image.shape[1], self.original_image.shape[0]))

            segmented_image_path = f"segmented_{method.lower()}.png"
            cv2.imwrite(segmented_image_path, segmented_image)
            self.segmented_image_path = segmented_image_path
            print(f"Segmentowany obraz zapisano jako: {segmented_image_path}")

            if method != "U-Net" and true_mask is not None:
                metrics = self.calculate_metrics(true_mask, segmented_image)
                self.report_data[method.lower()] = metrics
                self.last_metrics = metrics
                self.display_metrics(metrics)

            self.metrics_box.opacity = 1
            self.metrics_box.disabled = False

            self.save_experiment_to_db()
        else:
            print("Segmentacja nie powiodła się lub zwróciła wartość None.")

    def show_cluster_input_popup(self):
        try:
            original_img_texture = self.array_to_texture(self.original_image)
            self.original_image_display.texture = original_img_texture
        except Exception as e:
            print(f"Error resetting original image display: {e}")

        content = BoxLayout(orientation='vertical', padding=20, spacing=20)

        label = Label(
            text="Podaj liczbę klas dla metody Fuzzy C-means:",
            font_size=24,
            bold=True,
            size_hint=(1, None),
            height=40,
            halign="center",
            valign="middle",
        )
        content.add_widget(label)

        cluster_input = TextInput(
            text="3",
            multiline=False,
            input_filter="int",
            size_hint=(1, None),
            height=50,
        )
        content.add_widget(cluster_input)

        submit_button = Button(
            text="Uruchom",
            size_hint=(1, None),
            height=40,
            background_color=(0.2, 0.6, 0.8, 1),
            color=(1, 1, 1, 1),
        )
        submit_button.bind(on_press=lambda x: self.run_cmeans_segmentation(cluster_input.text))
        content.add_widget(submit_button)

        self.cluster_popup = Popup(
            title="Liczba klas dla C-means",
            content=content,
            size_hint=(0.3, 0.2),
        )
        self.cluster_popup.open()

    def run_cmeans_segmentation(self, n_clusters):
        try:
            n_clusters = int(n_clusters)
        except ValueError:
            print("Invalid number of clusters.")
            return

        if n_clusters <= 0:
            print("Number of clusters must be greater than zero.")
            return

        self.cluster_popup.dismiss()

        try:
            image_to_segment = self.original_image.copy()
            segmented_image, _, _ = segment_image_with_fuzzy_cmeans(self.selected_image_path, n_clusters=n_clusters)
        except Exception as e:
            print(f"Error in fuzzy C-means segmentation: {e}")
            return

        if not isinstance(segmented_image, np.ndarray):
            print("Fuzzy C-means segmentation did not return a valid numpy array.")
            return

        try:
            original_img_texture = self.array_to_texture(self.original_image)
            self.original_image_display.texture = original_img_texture
        except Exception as e:
            print(f"Error displaying original image: {e}")

        try:
            segmented_image = cv2.resize(segmented_image,
                                         (self.original_image.shape[1], self.original_image.shape[0]))
            self.segmented_image = segmented_image
        except cv2.error as e:
            print(f"Error resizing segmented image: {e}")
            return

        self.display_segmented_image(segmented_image)

        metrics = self.calculate_metrics(self.original_image, segmented_image)
        self.report_data['c-means'] = metrics
        self.last_metrics = metrics
        self.display_metrics(metrics)
        self.save_experiment_to_db()

    def display_original_image(self, img):
        try:
            original_img_texture = self.array_to_texture(img)
            self.original_image_display.texture = original_img_texture
            print("Original image displayed successfully.")
        except Exception as e:
            print(f"Error displaying original image: {e}")


    def display_segmented_image(self, img):
        try:
            if len(img.shape) == 2:
                colorfmt = 'luminance'
            elif len(img.shape) == 3:
                colorfmt = 'rgb'
            else:
                print("Unknown image format.")
                return

            img = np.flip(img, axis=0)
            texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt=colorfmt)
            texture.blit_buffer(img.tobytes(), colorfmt=colorfmt, bufferfmt='ubyte')
            self.segmented_image_display.texture = texture
            print("Segmented image displayed successfully.")
        except Exception as e:
            print(f"Error displaying segmented image: {e}")


    def display_metrics(self, metrics):
        self.metrics_box.clear_widgets()
        title_label = Label(
            text="[b]METRYKI OCENY[/b]",
            font_size=40,
            bold=True,
            markup=True,
            size_hint_y=None,
            height=60,
            halign="center",
            valign="middle",
            color=(1, 1, 1, 1)
        )
        title_label.bind(size=title_label.setter('text_size'))
        self.metrics_box.add_widget(title_label)

        selected_metrics = [
            ("Dokładność", "accuracy"),
            ("IOU", "iou"),
            ("Precyzja", "precision"),
            ("Czułość", "recall"),
            ("F1", "f1"),
            ("Dice", "dice"),
            ("Specyficzność", "specificity")
        ]
        selected_metrics = [
            (name, key) for name, key in selected_metrics if self.selected_metrics.get(name, False)
        ]

        if not selected_metrics:
            self.metrics_box.opacity = 0
            self.metrics_box.disabled = True
            return

        box_height = 80
        font_size = 30

        for name, key in selected_metrics:
            metric_value = metrics.get(key, None)
            if metric_value is None:
                label_text = f"[b]{name}: N/A[/b]"
            else:
                label_text = f"[b]{name}: {metric_value:.4f}[/b]"

            box = BoxLayout(orientation='horizontal', padding=10, size_hint_y=None, height=box_height)
            with box.canvas.before:
                Color(0.9, 0.9, 0.9, 1)
                RoundedRectangle(pos=box.pos, size=box.size, radius=[10])
            box.bind(pos=self.update_rounded_rectangle, size=self.update_rounded_rectangle)

            label = Label(
                text=label_text,
                font_size=font_size,
                markup=True,
                halign="center",
                valign="middle",
                color=(0, 0, 0, 1)
            )
            label.bind(size=label.setter('text_size'))

            box.add_widget(label)
            self.metrics_box.add_widget(box)

        self.metrics_box.opacity = 1
        self.metrics_box.disabled = False

    def run_unet_segmentation(self, image_path):
        model = load_model('unet_model_optimized.keras')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Image could not be loaded. Please check the path.")

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_size = image_gray.shape[:2]
        image_resized = cv2.resize(image_gray, (128, 128))
        image_resized = image_resized / 255.0
        image_resized = np.expand_dims(image_resized, axis=-1)
        image_resized = np.expand_dims(image_resized, axis=0)
        predicted_mask = model.predict(image_resized)[0]
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
        predicted_mask = cv2.resize(predicted_mask, (original_size[1], original_size[0]))

        return predicted_mask

    def normalize_mask_to_classes(self, mask, num_classes):
        if mask.max() > 1.0:
            mask = mask / 255.0

        thresholds = np.linspace(0, 1, num_classes + 1)
        class_map = np.zeros_like(mask, dtype=np.uint8)

        for class_idx in range(num_classes):
            class_map[(mask >= thresholds[class_idx]) & (mask < thresholds[class_idx + 1])] = class_idx

        return class_map

    def calculate_dice_score(self, true_mask, predicted_mask, num_classes):
        dice_scores = []
        for class_idx in range(num_classes):
            true_class = (true_mask == class_idx).astype(np.uint8)
            predicted_class = (predicted_mask == class_idx).astype(np.uint8)

            intersection = np.sum(true_class * predicted_class)
            size_true = np.sum(true_class)
            size_predicted = np.sum(predicted_class)
            dice = (2 * intersection) / (size_true + size_predicted) if (size_true + size_predicted) > 0 else 0
            dice_scores.append(dice)

        return np.mean(dice_scores)

    def calculate_metrics(self, true_mask, predicted_mask):
        num_classes = len(CLASS_COLORS)

        true_mask_normalized = self.normalize_mask_to_classes(true_mask, num_classes)
        predicted_mask_normalized = self.normalize_mask_to_classes(predicted_mask, num_classes)

        if true_mask_normalized.shape != predicted_mask_normalized.shape:
            predicted_mask_normalized = cv2.resize(
                predicted_mask_normalized,
                (true_mask_normalized.shape[1], true_mask_normalized.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        true_mask_flat = true_mask_normalized.flatten()
        predicted_mask_flat = predicted_mask_normalized.flatten()

        accuracy = accuracy_score(true_mask_flat, predicted_mask_flat)
        iou = jaccard_score(true_mask_flat, predicted_mask_flat, average="macro")
        precision = precision_score(true_mask_flat, predicted_mask_flat, average="macro")
        recall = recall_score(true_mask_flat, predicted_mask_flat, average="macro")
        f1 = f1_score(true_mask_flat, predicted_mask_flat, average="macro")
        dice = self.calculate_dice_score(true_mask_normalized, predicted_mask_normalized, num_classes)

        specificity_values = []
        for class_idx in range(num_classes):
            true_class = (true_mask_flat == class_idx).astype(np.uint8)
            pred_class = (predicted_mask_flat == class_idx).astype(np.uint8)

            tn = np.sum((true_class == 0) & (pred_class == 0))
            fp = np.sum((true_class == 0) & (pred_class == 1))

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_values.append(specificity)

        specificity = np.mean(specificity_values)

        return {
            "accuracy": accuracy,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "dice": dice,
            "specificity": specificity,
        }

    def run_comparison(self, instance):
        if not hasattr(self, 'selected_image_path'):
            self.accuracy_label.text = "Najpierw wybierz obraz!"
            return

        img = cv2.imread(self.selected_image_path, cv2.IMREAD_GRAYSCALE)

        otsu_result = self.apply_otsu_with_preprocessing(img)
        unet_result = self.run_unet_segmentation(self.selected_image_path)

        otsu_metrics = self.calculate_metrics(img, otsu_result)
        unet_metrics = self.calculate_metrics(img, unet_result)

        self.display_metrics(otsu_metrics)

        self.report_data = {
            'otsu': otsu_metrics,
            'unet': unet_metrics
        }

    def save_report(self, instance):
        if hasattr(self, 'report_data') and self.report_data:
            try:
                report_csv_path = "report.csv"
                fieldnames = ['method', 'accuracy', 'iou', 'precision', 'recall', 'f1', 'dice', 'specificity']
                with open(report_csv_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for method in self.report_data:
                        writer.writerow({'method': method.capitalize(), **self.report_data[method]})
                print(f"CSV report successfully saved at: {report_csv_path}")

                method = self.segment_button.text
                base_path = method.lower()
                visual_paths = self.generate_visualizations(base_path)
                report_path = f"raport_{method}.pdf"
                self.generate_pdf_report(method, visual_paths, report_path)

            except Exception as e:
                print(f"Error while saving report: {e}")

    def generate_visualizations(self, base_path):
        visual_paths = {}

        fig_hist = self.plot_intensity_histogram(self.original_image, "Histogram intensywnosci pikseli")
        hist_path = f"{base_path}_histogram.png"
        fig_hist.savefig(hist_path, bbox_inches='tight')
        plt.close(fig_hist)
        visual_paths['histogram'] = hist_path

        fig_overlay = self.plot_overlay(self.original_image, self.segmented_image, "Nakładka masek segmentacji")
        overlay_path = f"{base_path}_overlay.png"
        fig_overlay.savefig(overlay_path, bbox_inches='tight')
        plt.close(fig_overlay)
        visual_paths['overlay'] = overlay_path

        fig_area_dist = self.plot_area_distribution(self.segmented_image, "Rozkład obszarów segmentacji")
        area_dist_path = f"{base_path}_area_distribution.png"
        fig_area_dist.savefig(area_dist_path, bbox_inches='tight')
        plt.close(fig_area_dist)
        visual_paths['area_distribution'] = area_dist_path


        return visual_paths

    def save_texture_as_image(self, texture, output_path):
        if texture is None:
            print("No texture found to save.")
            return None

        size = texture.size
        pixels = texture.pixels
        img_data = np.frombuffer(pixels, dtype=np.uint8).reshape(size[1], size[0], 4)  # RGBA

        img_data_rgb = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)

        img_data_rgb = np.flip(img_data_rgb, axis=0)

        cv2.imwrite(output_path, cv2.cvtColor(img_data_rgb, cv2.COLOR_RGB2BGR))
        print(f"Image saved from texture: {output_path}")
        return output_path

    def generate_pdf_report(self, method, visual_paths, report_path):
        try:
            pdfmetrics.registerFont(TTFont('SpecialElite', 'SpecialElite-Regular.ttf'))

            report_number = len(self.database.fetch_all_experiments()) + 1
            current_date = dt.datetime.now().strftime("%Y-%m-%d")
            current_time = dt.datetime.now().strftime("%H:%M:%S")
            report_path = f"raport_{report_number}_{method}_{current_date}_{current_time.replace(':', '-')}.pdf"

            doc = SimpleDocTemplate(report_path, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30,
                                    bottomMargin=30)
            styles = getSampleStyleSheet()
            styles['Normal'].fontName = 'SpecialElite'
            styles['Title'].fontName = 'SpecialElite'
            styles['Heading3'].fontName = 'SpecialElite'
            elements = []

            title = Paragraph(f"<b>Raport Segmentacji - Metoda: {method}</b>", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 0.2 * inch))

            date_time_text = f"Data i godzina wygenerowania raportu: <b>{current_date} {current_time}</b>"
            elements.append(Paragraph(date_time_text, styles['Normal']))
            elements.append(Spacer(1, 0.2 * inch))

            intro_text = (
                f"Ten raport przedstawia wyniki segmentacji obrazu przy użyciu metody "
                f"<b>{method}</b>. Wyniki metryk, wizualizacje oraz analiza są zaprezentowane poniżej."
            )
            elements.append(Paragraph(intro_text, styles['Normal']))
            elements.append(Spacer(1, 0.2 * inch))

            metrics_data = [["Metryka", "Wartość"]]
            for metric, key in {
                "Accuracy": "accuracy",
                "IoU": "iou",
                "Precision": "precision",
                "Recall": "recall",
                "F1": "f1",
                "Dice": "dice",
                "Specificity": "specificity",
            }.items():
                value = self.report_data[method.lower()].get(key, "N/A")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                metrics_data.append([metric, value])

            table = Table(metrics_data, colWidths=[2.5 * inch, 2.5 * inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6c757d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'SpecialElite'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8f9fa'), colors.HexColor('#e9ecef')]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.5 * inch))

            pdf_segmented_image_path = f"segmented_{method.lower()}.png"
            self.save_texture_as_image(self.segmented_image_display.texture, pdf_segmented_image_path)

            images_data = [
                [
                    ReportLabImage(self.selected_image_path, width=2.2 * inch, height=2.2 * inch, hAlign='CENTER'),
                    ReportLabImage(self.selected_mask_path, width=2.2 * inch, height=2.2 * inch, hAlign='CENTER'),
                    ReportLabImage(pdf_segmented_image_path, width=2.2 * inch, height=2.2 * inch, hAlign='CENTER'),
                ],
                [
                    Paragraph("<b>Obraz oryginalny</b>", styles["Normal"]),
                    Paragraph("<b>Maska referencyjna</b>", styles["Normal"]),
                    Paragraph("<b>Obraz po segmentacji</b>", styles["Normal"]),
                ],
            ]

            images_table = Table(images_data, colWidths=[2.5 * inch, 2.5 * inch, 2.5 * inch])
            images_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ced4da')),
            ]))
            elements.append(images_table)
            elements.append(Spacer(1, 0.5 * inch))

            elements.append(PageBreak())

            gray_background_color = colors.HexColor('#f1f1f1')

            visualizations_title = Paragraph(f"<b>Wizualizacje</b>", styles['Heading3'])
            title_box = Table(
                [[visualizations_title]],
                colWidths=[6.5 * inch],
                style=[
                    ('BACKGROUND', (0, 0), (-1, -1), gray_background_color),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTSIZE', (0, 0), (-1, -1), 14),
                    ('FONTNAME', (0, 0), (-1, -1), 'SpecialElite'),
                ]
            )
            elements.append(title_box)
            elements.append(Spacer(1, 0.3 * inch))

            for label, path in visual_paths.items():
                image = ReportLabImage(path, width=5 * inch, height=3.5 * inch)
                visualization_box = Table(
                    [[image]],
                    colWidths=[6.5 * inch],
                    style=[
                        ('BACKGROUND', (0, 0), (-1, -1), gray_background_color),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]
                )
                elements.append(visualization_box)
                elements.append(Spacer(1, 0.3 * inch))

            doc.build(elements)
            print(f"PDF report saved at: {report_path}")

            success_popup = Popup(
                title="Raport wygenerowany",
                content=Label(text=f"Raport został wygenerowany i zapisany jako {report_path}.", font_size=24),
                size_hint=(0.4, 0.2),
            )
            success_popup.open()

        except Exception as e:
            print(f"Błąd podczas generowania raportu PDF: {e}")

    def show_segmentation_details(self, instance):
        metrics = self.calculate_metrics(self.original_image, self.segmented_image)

        content = BoxLayout(orientation='vertical', spacing=10, padding=[20, 20, 20, 20])
        with content.canvas.before:
            Color(0.95, 0.95, 0.95, 1)
            RoundedRectangle(size=content.size, pos=content.pos, radius=[15])
        content.bind(size=self.update_rounded_rectangle, pos=self.update_rounded_rectangle)

        table_layout = BoxLayout(orientation='vertical', spacing=5, size_hint=(1, None), height=400)

        header_layout = GridLayout(cols=2, size_hint=(1, None), height=40)
        with header_layout.canvas.before:
            Color(0.2, 0.4, 0.6, 1)
            RoundedRectangle(size=header_layout.size, pos=header_layout.pos, radius=[5])
        header_layout.bind(size=self.update_rounded_rectangle, pos=self.update_rounded_rectangle)
        header_layout.add_widget(Label(
            text="[b]Metryka[/b]", font_size=18, markup=True, halign="center", valign="middle", color=(1, 1, 1, 1)
        ))
        header_layout.add_widget(Label(
            text="[b]Wartość[/b]", font_size=18, markup=True, halign="center", valign="middle", color=(1, 1, 1, 1)
        ))
        table_layout.add_widget(header_layout)

        metric_names = {
            "Dokładność": metrics['accuracy'],
            "IOU": metrics['iou'],
            "Precyzja": metrics['precision'],
            "Czułość": metrics['recall'],
            "F1": metrics['f1'],
            "Dice": metrics['dice'],
            "Specyficzność": metrics['specificity']
        }

        for i, (name, value) in enumerate(metric_names.items()):
            row_color = (0.95, 0.95, 0.95, 1) if i % 2 == 0 else (0.85, 0.85, 0.85, 1)
            row_layout = GridLayout(cols=2, size_hint=(1, None), height=30)
            with row_layout.canvas.before:
                Color(*row_color)
                RoundedRectangle(size=row_layout.size, pos=row_layout.pos, radius=[5])
            row_layout.bind(size=self.update_rounded_rectangle, pos=self.update_rounded_rectangle)

            row_layout.add_widget(Label(
                text=f"[b]{name}[/b]", font_size=16, markup=True, size_hint=(1, None), height=30, halign="center",
                valign="middle", color=(0, 0, 0, 1)
            ))
            row_layout.add_widget(Label(
                text=f"{value:.4f}", font_size=16, size_hint=(1, None), height=30, halign="center", valign="middle",
                color=(0, 0, 0, 1)
            ))

            table_layout.add_widget(row_layout)

        content.add_widget(table_layout)

        spacer = BoxLayout(size_hint=(1, None), height=10)
        content.add_widget(spacer)

        chart_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, 1))

        intensity_hist_img = self.save_and_load_chart_as_image(
            self.plot_intensity_histogram(self.original_image, "Histogram intensywności pikseli"))
        overlay_img = self.save_and_load_chart_as_image(
            self.plot_overlay(self.original_image, self.segmented_image, "Nakładka masek segmentacji"))
        area_dist_img = self.save_and_load_chart_as_image(
            self.plot_area_distribution(self.segmented_image, "Rozkład obszarów segmentacji"))

        intensity_hist_img.size_hint = (1 / 3, 0.8)
        overlay_img.size_hint = (1 / 3, 0.8)
        area_dist_img.size_hint = (1 / 3, 0.8)

        chart_layout.add_widget(intensity_hist_img)
        chart_layout.add_widget(overlay_img)
        chart_layout.add_widget(area_dist_img)

        content.add_widget(chart_layout)

        close_button = Button(
            text="Zamknij",
            size_hint=(1, None),
            height=50,
            font_size=20,
            bold=True,
            background_color=(0.2, 0.5, 0.8, 1),
            color=(1, 1, 1, 1)
        )
        close_button.bind(on_release=lambda x: self.details_popup.dismiss())
        content.add_widget(close_button)

        self.details_popup = Popup(
            title="Szczegóły segmentacji",
            content=content,
            size_hint=(0.85, 0.85)
        )
        self.details_popup.open()


    def plot_intensity_histogram(self, image, title):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.hist(image.ravel(), bins=256, color='gray')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Intensywność pikseli')
        ax.set_ylabel('Częstotliwość')
        return fig

    def plot_overlay(self, original, mask, title):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.imshow(original, cmap='gray')
        ax.imshow(mask, cmap='jet', alpha=0.5)
        ax.set_title(title, fontsize=16)
        return fig

    def plot_area_distribution(self, mask, title):
        labels, counts = np.unique(mask, return_counts=True)
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.bar(labels, counts, color=['black', 'red'])
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Etykieta')
        ax.set_ylabel('Liczba pikseli')
        return fig


    def save_and_load_chart_as_image(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return Image(texture=CoreImage(buf, ext='png').texture)

    def save_experiment_to_db(self):
        if not hasattr(self, 'last_metrics'):
            print("No metrics available to save.")
            return

        method = self.segment_button.text
        if method == "Wybierz metodę segmentacji":
            print("No segmentation method selected.")
            return

        segmented_image_path = f"segmented_{method.lower()}.png"
        if hasattr(self, 'segmented_image') and self.segmented_image is not None:
            cv2.imwrite(segmented_image_path, self.segmented_image)

        metrics = self.last_metrics
        print("Saving experiment to DB with the following details:")
        print(f"Method: {method}")
        print(f"Original Image Path: {self.selected_image_path}")
        print(f"Mask Path: {getattr(self, 'selected_mask_path', None)}")
        print(f"Segmented Image Path: {segmented_image_path}")
        print(f"Metrics: {metrics}")

        processing_time = float(self.processing_time_label.text.split(':')[-1].strip().split()[0])

        try:
            self.database.save_experiment(
                method=method,
                original_image_path=self.selected_image_path,
                mask_path=getattr(self, 'selected_mask_path', None),
                segmented_image_path=segmented_image_path,
                metrics=metrics,
                segmentation_time=processing_time
            )
            print("Experiment saved successfully.")
        except Exception as e:
            print(f"Error saving experiment to database: {e}")

    def show_database_view(self, instance):
        experiments = self.database.fetch_all_experiments()

        if not experiments:
            self.show_popup("Baza danych", "Brak zapisanych eksperymentów.")
            return

        row_data = self.format_experiments_data(experiments)

        column_data = [
            ("ID", dp(30)),
            ("Metoda", dp(40)),
            ("Data", dp(70)),
            ("Accuracy", dp(30)),
            ("IoU", dp(30)),
            ("Precision", dp(30)),
            ("Recall", dp(30)),
            ("F1", dp(30)),
            ("Dice", dp(30)),
            ("Specificity", dp(30)),
            ("Czas Segmentacji (s)", dp(40))
        ]

        data_table = MDDataTable(
            size_hint=(1, 1),
            use_pagination=True,
            rows_num=10,
            column_data=column_data,
            row_data=row_data,
        )

        main_layout = BoxLayout(orientation="vertical", spacing=20, padding=[20, 20, 20, 20])
        main_layout.add_widget(self.create_table_with_buttons(data_table))
        main_layout.add_widget(self.create_close_button_layout())

        self.db_view_popup = Popup(
            title="Baza danych eksperymentów",
            content=main_layout,
            size_hint=(0.95, 0.95),
        )
        self.db_view_popup.open()

    def format_experiments_data(self, experiments):
        def safe_float(value, default=0.0):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        return [
            (
                str(exp["id"]),
                exp["Method"],
                exp["timestamp"],
                f"{safe_float(exp.get('Accuracy')):.4f}",
                f"{safe_float(exp.get('IoU')):.4f}",
                f"{safe_float(exp.get('Precision')):.4f}",
                f"{safe_float(exp.get('Recall')):.4f}",
                f"{safe_float(exp.get('F1')):.4f}",
                f"{safe_float(exp.get('Dice')):.4f}",
                f"{safe_float(exp.get('Specificity')):.4f}",
                f"{safe_float(exp.get('SegmentationTime')):.2f}"
            )
            for exp in experiments
        ]

    def create_table_with_buttons(self, data_table):
        table_layout = BoxLayout(orientation="horizontal", spacing=20, size_hint=(1, 1))
        table_layout.add_widget(data_table)

        button_column = BoxLayout(orientation="vertical", spacing=40, size_hint=(None, 1), width=200)
        button_column.add_widget(Widget(size_hint=(1, 0.2)))
        button_column.add_widget(self.create_icon_button("csv_2.png", self.export_csv))
        button_column.add_widget(self.create_icon_button("xls_2.png", self.export_xls))
        button_column.add_widget(Widget(size_hint=(1, 0.2)))

        table_layout.add_widget(button_column)
        return table_layout

    def create_icon_button(self, icon_path, callback):
        return Button(
            size_hint=(None, None),
            size=(160, 160),
            background_normal=icon_path,
            background_down=icon_path,
            border=(20, 20, 20, 20),
            on_press=callback,
        )

    def create_close_button_layout(self):
        close_button_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=80)
        close_button = Button(
            text="Zamknij",
            size_hint=(None, None),
            size=(200, 80),
            font_size=28,
            bold=True,
            background_color=(0.2, 0.5, 0.8, 1),
            color=(1, 1, 1, 1),
            on_release=lambda x: self.db_view_popup.dismiss(),
        )

        close_button_layout.add_widget(Widget(size_hint=(1, 1)))
        close_button_layout.add_widget(close_button)
        close_button_layout.add_widget(Widget(size_hint=(1, 1)))

        return close_button_layout

    def show_popup(self, title, message):
        popup = Popup(
            title=title,
            content=Label(text=message, font_size=30),
            size_hint=(0.9, 0.5),
        )
        popup.open()

    def fetch_saved_experiments(self):
        experiments = self.database.fetch_all_experiments()
        return experiments

    def export_csv(self, instance):
        experiments = self.database.fetch_all_experiments()

        data = [
            {
                'ID': exp["id"],
                'Method': exp["Method"],
                'Timestamp': exp["timestamp"],
                'Accuracy': exp.get('Accuracy', 0),
                'IoU': exp.get('IoU', 0),
                'Precision': exp.get('Precision', 0),
                'Recall': exp.get('Recall', 0),
                'F1': exp.get('F1', 0),
                'Dice': exp.get('Dice', 0),
                'Specificity': exp.get('Specificity', 0)
            }
            for exp in experiments
        ]

        df = pd.DataFrame(data)

        csv_file = "experiment_data.csv"
        df.to_csv(csv_file, index=False)
        print(f"Data exported to {csv_file}")

        success_popup = Popup(
            title="Eksport zakończony",
            content=Label(text="Dane zostały wyeksportowane do CSV."),
            size_hint=(0.5, 0.3)
        )
        success_popup.open()

    def export_xls(self, instance):
        experiments = self.database.fetch_all_experiments()

        data = [
            {
                'ID': exp["id"],
                'Method': exp["Method"],
                'Timestamp': exp["timestamp"],
                'Accuracy': exp.get('Accuracy', 0),
                'IoU': exp.get('IoU', 0),
                'Precision': exp.get('Precision', 0),
                'Recall': exp.get('Recall', 0),
                'F1': exp.get('F1', 0),
                'Dice': exp.get('Dice', 0),
                'Specificity': exp.get('Specificity', 0)
            }
            for exp in experiments
        ]

        df = pd.DataFrame(data)

        excel_file = "experiment_data.xlsx"
        df.to_excel(excel_file, index=False)
        print(f"Data exported to {excel_file}")

        success_popup = Popup(
            title="Eksport zakończony",
            content=Label(text="Dane zostały wyeksportowane do XLS."),
            size_hint=(0.5, 0.3)
        )
        success_popup.open()

    def save_segmented_image(self, method, default_name=None):
        from pathlib import Path
        desktop = Path.home() / "Desktop"
        results_dir = desktop / "Segmentacja" / "Wyniki"
        results_dir.mkdir(parents=True, exist_ok=True)

        if not default_name:
            default_name = f"{method.lower()}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        file_path = results_dir / default_name

        if self.segmented_image_display.texture is not None:
            texture = self.segmented_image_display.texture
            size = texture.size
            img_data = texture.pixels
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(size[1], size[0], 4)  # Convert to NumPy array

            img_rgb = img[:, :, :3]

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(file_path), img_bgr)
            print(f"Segmented image saved at: {file_path}")
            return file_path
        else:
            print("No image to save.")
            return None

    def show_save_popup(self, method, default_name):
        content = BoxLayout(orientation="vertical", spacing=20, padding=[20, 20, 20, 20])
        with content.canvas.before:
            Color(0.95, 0.95, 0.95, 1)
            RoundedRectangle(size=content.size, pos=content.pos, radius=[15])
        content.bind(size=self.update_rounded_rectangle, pos=self.update_rounded_rectangle)

        label = Label(
            text="Wprowadź nazwę pliku lub użyj domyślnej:",
            font_size=26,
            halign="center",
            valign="middle",
            bold=True,
            size_hint=(1, None),
            height=40,
            color=(0, 0, 0, 1)
        )
        content.add_widget(label)

        file_name_input = TextInput(
            text=default_name,
            multiline=False,
            size_hint=(1, None),
            height=50,
        )
        content.add_widget(file_name_input)

        save_button = Button(
            text="Zapisz obraz",
            size_hint=(1, None),
            height=60,
            font_size=26,
            bold=True,
            background_color=(0.5, 0.5, 0.5, 1),
            color=(1, 1, 1, 1)
        )
        save_button.bind(on_press=lambda x: self.save_image_with_name(file_name_input.text, method))
        content.add_widget(save_button)

        close_button = Button(
            text="Zamknij",
            size_hint=(1, None),
            height=60,
            font_size=26,
            bold=True,
            background_color=(0.1, 0.3, 0.7, 1),
            color=(1, 1, 1, 1)
        )
        close_button.bind(on_release=lambda x: self.save_popup.dismiss())
        content.add_widget(close_button)

        self.save_popup = Popup(
            title="Zapis obrazu po segmentacji",
            content=content,
            size_hint=(0.5, 0.3)
        )
        self.save_popup.open()

    def save_image_with_name(self, file_name, method):
        desktop = Path.home() / "Desktop"
        results_dir = desktop / "Segmentacja" / "Wyniki"
        results_dir.mkdir(parents=True, exist_ok=True)

        if not file_name.endswith(".png"):
            file_name += ".png"

        file_path = results_dir / file_name

        texture = self.segmented_image_display.texture
        if texture:
            saved_path = self.save_texture_as_image(texture, str(file_path))
            if saved_path:
                print(f"Obraz zapisano jako: {saved_path}")

                self.save_popup.dismiss()

                success_popup = Popup(
                    title="Zapisano obraz",
                    content=Label(text=f"Obraz został zapisany w: {saved_path}", font_size=20),
                    size_hint=(0.5, 0.2)
                )
                success_popup.open()
            else:
                print("Błąd podczas zapisywania obrazu.")
        else:
            print("Nie znaleziono tekstury do zapisania.")

    def update(self, dt):
        pass


if __name__ == '__main__':
    MainApp().run()