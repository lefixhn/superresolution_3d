import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Button, VBox, HBox, Output
from IPython.display import display

current_file_index = 0

def view_h5_files(folder_path):
    global current_file_index
    current_file_index = 0

    try:
        h5_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])
        if not h5_files:
            print(f"Keine .h5-Dateien im Ordner gefunden: {folder_path}")
            return
    except FileNotFoundError:
        print(f"Fehler: Der Ordner wurde nicht gefunden: {folder_path}")
        return

    # --- ÄNDERUNG 1: Erstelle die Plot-Leinwand nur EINMAL am Anfang ---
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.close(fig) # Verhindert, dass die leere Figur sofort angezeigt wird

    slice_slider = IntSlider(min=0, max=100, step=1, value=0, description='Schicht:', continuous_update=False)
    next_button = Button(description="Nächstes Bild", button_style='success')
    out = Output()

    # --- ÄNDERUNG 2: Die plot_slice Funktion erhält die Achse (ax) als Argument ---
    def plot_slice(ax, file_path, slice_idx):
        try:
            with h5py.File(file_path, 'r') as hf:
                if 'reconstruction_rss' in hf.keys():
                    volume = hf['reconstruction_rss'][:]
                elif 'reconstruction_esc' in hf.keys():
                    volume = hf['reconstruction_esc'][:]
                else:
                    print(f"Konnte keinen passenden Datensatz in {os.path.basename(file_path)} finden.")
                    return

                slice_data = volume[slice_idx]
                if slice_data.max() > 0:
                    slice_data = slice_data / slice_data.max()

                # --- ÄNDERUNG 3: Leinwand leeren, statt eine neue zu erstellen ---
                ax.clear()
                ax.imshow(slice_data, cmap='gray')
                ax.set_title(f"{os.path.basename(file_path)}\nSchicht: {slice_idx + 1}/{volume.shape[0]}")
                ax.axis('off')
                
                # Die Figur wird nun innerhalb des Output-Widgets neu gezeichnet
                with out:
                    out.clear_output(wait=True)
                    display(fig)

        except Exception as e:
            print(f"Fehler beim Laden oder Anzeigen der Datei {os.path.basename(file_path)}: {e}")

    def update_plot(file_index):
        file_path = os.path.join(folder_path, h5_files[file_index])
        try:
            with h5py.File(file_path, 'r') as hf:
                if 'reconstruction_rss' in hf.keys():
                    num_slices = hf['reconstruction_rss'].shape[0]
                else:
                    num_slices = hf['reconstruction_esc'].shape[0]

                middle_slice = num_slices // 2
                slice_slider.max = num_slices - 1
                slice_slider.value = middle_slice
                # --- ÄNDERUNG 4: Übergebe 'ax' an die Plot-Funktion ---
                plot_slice(ax, file_path, middle_slice)
        except Exception as e:
            print(f"Fehler beim Laden der Datei {os.path.basename(file_path)}: {e}")

    def on_slice_change(change):
        file_path = os.path.join(folder_path, h5_files[current_file_index])
        # --- ÄNDERUNG 5: Übergebe 'ax' an die Plot-Funktion ---
        plot_slice(ax, file_path, change.new)

    def on_button_click(b):
        global current_file_index
        current_file_index = (current_file_index + 1) % len(h5_files)
        update_plot(current_file_index)

    slice_slider.observe(on_slice_change, names='value')
    next_button.on_click(on_button_click)
    
    # Zeige die Steuerelemente und den Plot-Bereich an
    display(VBox([HBox([slice_slider, next_button]), out]))
    
    update_plot(0)