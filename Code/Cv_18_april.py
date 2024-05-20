"""
18th april. 2024

"""

import argparse
import os
import pandas as pd
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, confusion_matrix,roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from itertools import cycle
from tensorflow.keras import layers



def train_k_fold_model(fold, train_batches, valid_batches, num_classes, model_checkpoints,ReduceLR,EarlyStop, experiment_name,df):
    # Construir modelo
    model = build_model(num_classes=num_classes)

    # Ajustar el cálculo de pesos de clase para las clases presentes en el conjunto de entrenamiento
    label_encoder = LabelEncoder()
    train_classes_encoded = label_encoder.fit_transform(train_batches.classes)
    class_weights_train = class_weight.compute_class_weight(
        'balanced', classes=np.unique(train_classes_encoded), y=train_classes_encoded)
    class_weights_dict_train = dict(zip(np.unique(train_classes_encoded), class_weights_train))

    # Entrenar el modelo inicialmente (con capas congeladas)
    print(f"Training initial phase for Fold {fold+1} with no* frozen layers for {args.num_initial_epochs} epochs.")
    history_initial = model.fit(
        train_batches,
        validation_data=valid_batches,
        epochs=args.num_initial_epochs,
        callbacks=[model_checkpoints[fold], ReduceLR, EarlyStop],
        class_weight=class_weights_dict_train,
       workers=4
    )

    # Descongelar las capas para el ajuste fino y continuar el entrenamiento
    unfreeze_model(model)
    print(f"Starting fine-tuning phase for Fold {fold+1} for additional {args.total_epochs - args.num_initial_epochs} epochs.")
    history_finetune = model.fit(
        train_batches,
        validation_data=valid_batches,
        epochs=args.total_epochs,
        callbacks=[model_checkpoints[fold], ReduceLR, EarlyStop],
        class_weight=class_weights_dict_train,
        workers=4,
        initial_epoch=args.num_initial_epochs)
    

    # Combinar las historias de entrenamiento inicial y ajuste fino para análisis
    # import pdb;pdb.set_trace()
    combined_history = {
        'accuracy': history_initial.history['accuracy'] + history_finetune.history['accuracy'],
        'val_accuracy': history_initial.history['val_accuracy'] + history_finetune.history['val_accuracy'],
        'loss': history_initial.history['loss'] + history_finetune.history['loss'],
        'val_loss': history_initial.history['val_loss'] + history_finetune.history['val_loss']
    }
    plt.figure() # SEBASTIAN: revisar inicializacion con tamaño predeterminado
    # Plot training history y guardar la imagen
    plt.plot(combined_history['accuracy'], label='Training Accuracy')
    plt.plot(combined_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Crear la carpeta si no existe
    experiment_folder = f"{experiment_name}/fold_{fold + 1}"
    os.makedirs(experiment_folder, exist_ok=True)

    # Guardar la imagen
    plt.savefig(f'{experiment_folder}/training_validation_plot.png')
    plt.show()

    # Crear una segunda figura para la gráfica de pérdida
    plt.figure()

    # Plot training y validation loss y guardar la imagen
    plt.plot(combined_history['loss'], label='Training Loss')
    plt.plot(combined_history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Guardar la imagen
    plt.savefig(f'{experiment_folder}/loss_plot.png')
    plt.show()
    # CURVA ROC
    # Evaluación del modelo en los datos de validación
    val_loss, val_accuracy = model.evaluate(valid_batches)
    print(f"Validation Accuracy for Fold {fold + 1}: {val_accuracy}")

    # Predicciones del modelo en los datos de validación
    y_pred = model.predict(valid_batches)

    # Convertir las etiquetas verdaderas a formato binario
    y_true = label_binarize(valid_batches.classes, classes=np.arange(len(df['class'].unique())))

    # Calcular las curvas ROC y el área bajo la curva (AUC) para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(train_classes_encoded))):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Trazar las curvas ROC para cada clase
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta'])
    for i, color in zip(range(len(np.unique(train_classes_encoded))), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve (class {i}) (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for each class')
    plt.legend(loc='lower right')

    # Guardar la imagen
    plt.savefig(f'{experiment_folder}/roc_curve_plot.png')
    plt.show()

    # CALCULAR MÉTRICAS
    # Extraer testing data
    val_predictions_enc = model.predict(valid_batches)
    val_predictions = np.argmax(val_predictions_enc, axis=-1)

    # Extraer true data
    val_data = []
    val_labels = []
    for i in range(len(valid_batches)):
        batch = valid_batches[i]
        val_data.append(batch[0])
        val_labels.append(batch[1])

    val_data = np.concatenate(val_data)
    val_labels_enc = np.concatenate(val_labels)
    val_labels = np.argmax(val_labels_enc, axis=1)

    # Evaluar el modelo en el conjunto de prueba
    val_loss, val_accuracy = model.evaluate(valid_batches, verbose=1)
    #val_loss, val_accuracy = model.evaluate(valid_batches)
    val_accuracy_percent = val_accuracy * 100

    # Calcular la matriz de confusión
    fold_confusion_matrix = confusion_matrix(val_labels, val_predictions)

    # Guardar la matriz de confusión en un archivo CSV
    confusion_matrix_df = pd.DataFrame(fold_confusion_matrix, columns=df['class'].unique(), index=df['class'].unique())
    confusion_matrix_df.to_csv(f'{experiment_folder}/confusion_matrix.csv')

    # Calcular y guardar las métricas de la matriz de confusión por cada clase en un archivo CSV
    class_metrics_data = {
        "Class": df['class'].unique(),
        "Precision": precision_score(val_labels, val_predictions, average=None),
        "Recall": recall_score(val_labels, val_predictions, average=None),
        "F1 Score": f1_score(val_labels, val_predictions, average=None)
    }
    class_metrics_df = pd.DataFrame(class_metrics_data)
    class_metrics_df.to_csv(f'{experiment_folder}/class_metrics.csv', index=False)

    # Calculate precision, recall, y specificity (micro-averaged)
    precision = precision_score(val_labels, val_predictions, average='micro')
    recall = recall_score(val_labels, val_predictions, average='micro')

    # Calculate true negatives, false positives, y specificity (micro-averaged)
    tn = np.sum((val_labels != 1) & (val_predictions != 1))
    fp = np.sum((val_labels != 1) & (val_predictions == 1))
    specificity = tn / (tn + fp)

    # Calculate F1 score (weighted average)
    f1 = f1_score(val_labels, val_predictions, average='weighted')

    # Additional details
    print("-" * 20)
    print("Additional Details:")
    print(f"Test Predictions Shape: {val_predictions.shape}")
    print(f"Test Labels Shape: {val_labels.shape}")
    print(f"F1 Score: {f1}")

    # Define your data as a dictionary
    data = {
        "Test Loss": [val_loss],
        "Test Accuracy (%)": [val_accuracy_percent],
        "F1 Score": [f1],
        "Precision": [precision],
        "Recall": [recall],
        "Specificity": [specificity]
    }

    # Create a DataFrame from the dictionary
    df_metrics = pd.DataFrame(data)

    # Guardar métricas en un archivo CSV
    df_metrics.to_csv(f'{experiment_folder}/metrics.csv', index=False)

    return df_metrics, fold_confusion_matrix  # Devuelve las métricas y la matriz de confusión)


def build_model(num_classes, im_size=(224, 224, 3), num_layers_unfreeze=50):
    base_model = EfficientNetB3(include_top=False, weights="imagenet", input_shape=im_size)

    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation="softmax", name="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def unfreeze_model(model, num_layers_unfreeze=50):
    # We unfreeze the top N layers while leaving BatchNorm layers frozen
    # num_layers_unfreeze puede ser pasado como argumento o, si está almacenado en el modelo, usado directamente.
    for layer in model.layers[-num_layers_unfreeze:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def get_args_parser():
    
    parser = argparse.ArgumentParser('E-Net', add_help=False)
        
    # Default values from the provided variables
    parser.add_argument('--experiment_name', default="BS_64_EPOCHS_CV_9k", type=str,
                        help='Name of the experiment, creates a folder with the same name.')
    parser.add_argument('--im_size', default=(224, 224, 3), type=tuple,
                        help='Image size')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--num_folds', default=5, type=int,
                        help='Number of folds')
    parser.add_argument('--num_initial_epochs', default=50, type=int,
                        help='Number of initial epochs')
    parser.add_argument('--total_epochs', default=125, type=int,
                        help='Total number of epochs')
    parser.add_argument('--data_path', default='/media/enc/vera1/sebastian/data/9k_urbanphony/Data_set_9k', type=str,
                        help='Path to the folder containing images and subfolders')
    parser.add_argument('--csv_file', default='/media/enc/vera1/sebastian/data/9k_urbanphony/Data_9k.csv', type=str,
                        help='Path to the CSV file')
    parser.add_argument('--folds', default=5, type=int,
                        help='folds')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed')
    
    return parser
    

def main(args):
    # SEB: set seed for reproducibility. 
    seed = int(args.seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Print available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Available GPUs:".center(60, "-"))
        for gpu in gpus:
            print("Name:", gpu.name, "  Type:", gpu.device_type)
    else:
        print("No GPUs available.")
    print("{}".format(args).replace(', ', ',\n'))

    # Listas para almacenar nombres de imágenes y etiquetas
    nombres_imagenes = []
    etiquetas = []

    # Recorrer la carpeta y todas sus subcarpetas de manera recursiva
    for raiz, subcarpetas, archivos in os.walk(args.data_path):
        for nombre_archivo in archivos:
            if nombre_archivo.endswith(('.png')):  # Ajustar las extensiones según tus archivos
                # Extraer la etiqueta del nombre de la carpeta
                etiqueta = os.path.basename(raiz)
                nombres_imagenes.append(nombre_archivo)
                etiquetas.append(etiqueta)

    # Crear un Marco de Datos con los datos recopilados
    df = pd.DataFrame({'name': nombres_imagenes, 'class': etiquetas})

    # Guardar el DataFrame en un archivo CSV

    df.to_csv(args.csv_file, index=False)

    print(f"Archivo CSV generado y guardado en {args.csv_file}")

    os.makedirs(args.experiment_name, exist_ok=True)

    # Load DataFrame from CSV
    df = pd.read_csv(args.csv_file)

    df['image_path'] = f'{args.data_path}'+ '/'  + df['class'] + '/' + df['name']  # Update this line

    # Codificar las clases con LabelEncoder
    label_encoder = LabelEncoder()
    df['class_encoded'] = label_encoder.fit_transform(df['class'])

    saving_model = f"{args.experiment_name}/models"
    os.makedirs(saving_model, exist_ok = True)

    # Imprimir la cantidad de datos y clases
    print(f"Number of data samples: {len(df)}")
    print(f"Number of unique classes: {len(df['class'].unique())}")

    # Create data generators
    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.1,
        fill_mode="nearest",
    )

    # Imprimir algunas rutas de imágenes para verificar
    print("Sample Image Paths:")
    print(df['image_path'].head())

    """## Finetuning B3 EfficientNet

    https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#transfer-learning-from-pretrained-weights
    """

    # Estimate class weights for unbalanced dataset using the classes from DataFrame 'df'
    classes_unique = np.unique(df['class'])
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=classes_unique, y=df['class']
    )
    class_weights_dict = dict(zip(classes_unique, class_weights))

    # ModelCheckpoint callback
    model_checkpoints = []

    for fold in range(args.num_folds):
        model_checkpoint = ModelCheckpoint(
            f"{saving_model}/{args.experiment_name}_model_fold_{fold + 1}.weights.h5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            save_freq='epoch',
        )
        model_checkpoints.append(model_checkpoint)

    # ReduceLR callback
    ReduceLR = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=3e-4
    )

    # EarlyStopping callback
    EarlyStop = EarlyStopping(
        monitor="val_loss", patience=10, verbose=1, mode="auto"
    )

    # Build and compile the model outside the k-fold loop
    model = build_model(num_classes=len(df['class'].unique()))

    # Stratified K-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    lista_metrics = []
    final_confusion_matrix = np.zeros((len(df['class'].unique()), len(df['class'].unique())))

    # Lista para almacenar DataFrames de métricas de clases de cada pliegue
    class_metrics_dfs = []

    # Imprimir información sobre las particiones de entrenamiento y validación
    for fold, (train_index, val_index) in enumerate(kf.split(df['image_path'], df['class_encoded'])):
        print(f"\nFold {fold + 1}/{args.num_folds}")
        print(f"Number of training samples: {len(train_index)}")
        print(f"Number of validation samples: {len(val_index)}")

        train_df_fold = df.iloc[train_index]
        val_df_fold = df.iloc[val_index]

        # Print unique classes in training and validation sets (optional)
        print(f"Unique classes in training set: {train_df_fold['class'].unique()}")
        print(f"Unique classes in validation set: {val_df_fold['class'].unique()}")

        # Print some sample image paths to debug (optional)
        print("Sample Training Image Paths:")
        print(train_df_fold['image_path'].head())

        print("Sample Validation Image Paths:")
        print(val_df_fold['image_path'].head())

        train_batches = train_datagen.flow_from_dataframe(
            train_df_fold,
            directory=None,
            x_col='image_path',
            y_col='class',
            target_size=args.im_size[:2],
            class_mode="categorical",
            shuffle=True,
            batch_size=args.batch_size,
        )

        valid_batches = valid_datagen.flow_from_dataframe(
            val_df_fold,
            directory=None,
            x_col='image_path',
            y_col='class',
            target_size=args.im_size[:2],
            class_mode="categorical",
            shuffle=False,
            batch_size=args.batch_size,
        )

        # Call train_k_fold_model with all required arguments
        df_metrics, fold_confusion_matrix = train_k_fold_model(fold, train_batches, valid_batches, len(df['class'].unique()), model_checkpoints,ReduceLR,EarlyStop, args.experiment_name, df)
        lista_metrics.append(df_metrics)

        # Acumular la matriz de confusión para cada fold
        final_confusion_matrix += fold_confusion_matrix

        # Crear la carpeta si no existe
        experiment_folder = f"{args.experiment_name}/fold_{fold + 1}"
        os.makedirs(experiment_folder, exist_ok=True)

        # Guardar la matriz de confusión del fold
        np.save(f'{experiment_folder}/confusion_matrix.npy', fold_confusion_matrix)

        # Calcular el promedio de las métricas por clase para el fold actual
        class_metrics_path = f"{experiment_folder}/class_metrics.csv"
        class_metrics_df = pd.read_csv(class_metrics_path)
        class_metrics_dfs.append(class_metrics_df)

        average_class_metrics_fold = class_metrics_df.groupby('Class').mean().reset_index()
        print(f"\nAverage Class Metrics for Fold {fold + 1}:\n{average_class_metrics_fold}")

    # Calcular el promedio general de las métricas por clase
    average_class_metrics_overall = pd.concat(class_metrics_dfs).groupby('Class').mean().reset_index()
    print("\nOverall Average Class Metrics:\n", average_class_metrics_overall)

    # Extraer métricas de la última iteración
    df_metrics = lista_metrics[-1]

    # Crear la carpeta si no existe
    os.makedirs(args.experiment_name, exist_ok=True)

    # Calcular métricas promedio
    resultados_promedio_metrics = pd.concat(lista_metrics).groupby(level=0).mean()

    # Mostrar métricas promedio
    print("\nAverage Metrics:")
    print(resultados_promedio_metrics)

    # Guardar el promedio general de las métricas por clase en un archivo CSV
    average_class_metrics_overall.to_csv(f'{args.experiment_name}/average_class_metrics_overall.csv', index=False)
    print(f"\nOverall Average Class Metrics saved to {args.experiment_name}/average_class_metrics_overall.csv")

    # Mostrar la matriz de confusión final
    plt.figure(figsize=(10, 8))
    sns.heatmap(final_confusion_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=df['class'].unique(), yticklabels=df['class'].unique())
    plt.title('Final Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Guardar la matriz de confusión final
    plt.savefig(f'{args.experiment_name}/final_confusion_matrix.png')

    plt.show()

    # Calculate average metrics directly
    average_accuracy = np.mean([df['Test Accuracy (%)'].mean() for df in lista_metrics])
    average_f1_score = np.mean([df['F1 Score'].mean() for df in lista_metrics])
    average_precision = np.mean([df['Precision'].mean() for df in lista_metrics])
    average_recall = np.mean([df['Recall'].mean() for df in lista_metrics])
    average_specificity = np.mean([df['Specificity'].mean() for df in lista_metrics])

    # Create a DataFrame with average metrics
    average_metrics_data = {
        "Average Accuracy": [average_accuracy],
        "Average F1 Score": [average_f1_score],
        "Average Precision": [average_precision],
        "Average Recall": [average_recall],
        "Average Specificity": [average_specificity]
        }

    average_metrics_df = pd.DataFrame(average_metrics_data)

    # Guardar el DataFrame en un archivo CSV dentro de la carpeta específica
    average_metrics_csv_path = f'{args.experiment_name}/average_metrics.csv'
    average_metrics_df.to_csv(average_metrics_csv_path, index=False)

    # Mostrar y guardar el DataFrame
    print("Average Metrics:")
    print(average_metrics_df)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
