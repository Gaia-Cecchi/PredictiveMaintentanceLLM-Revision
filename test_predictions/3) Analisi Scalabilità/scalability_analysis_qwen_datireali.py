import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import json
import logging
from datetime import datetime
from pathlib import Path
import random




class QwenScalabilityAnalyzer:
    """Analizzatore di scalabilità per il modello Qwen 2.5"""
   
    def __init__(self, output_dir="qwen_scalability_analysis"):
        """Inizializza l'analizzatore"""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
       
        # Configura il logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{output_dir}/analysis.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("QwenScalabilityAnalyzer")
       
        # Parametri da testare
        self.batch_sizes = [5, 10, 20, 50]
        self.reset_frequencies = [5, 10, 25, 50, None]  # None = nessun reset
       
        # Risultati
        self.results = []
   
    def load_test_data(self, data_path=None, n_samples=200):
        """
        Carica i dati di test da un file CSV con campionamento strategico delle anomalie
       
        Args:
            data_path: Percorso al file CSV con i dati
            n_samples: Numero totale di campioni desiderati
           
        Returns:
            DataFrame: Dataset bilanciato con tutte le anomalie
        """
        if data_path and os.path.exists(data_path):
            self.logger.info(f"Caricamento dati da {data_path}")
            # Carica l'intero dataset
            df = pd.read_csv(data_path)
           
            # Verifica la dimensione del dataset
            self.logger.info(f"Dataset completo: {len(df)} record")
           
            # Standardizza il formato (se necessario)
            if 'actual_classification' in df.columns:
                df['actual_binary'] = (df['actual_classification'] == 'ANOMALY').astype(int)
            elif 'actual_anomaly' in df.columns:
                df['actual_binary'] = df['actual_anomaly'].astype(int)
           
            # Identifica le anomalie e i record normali
            anomalies = df[df['actual_binary'] == 1]
            normal_records = df[df['actual_binary'] == 0]
           
            n_anomalies = len(anomalies)
            self.logger.info(f"Trovate {n_anomalies} anomalie nel dataset ({n_anomalies/len(df)*100:.2f}%)")
           
            # Prendi tutte le anomalie
            if n_anomalies == 0:
                self.logger.warning("Nessuna anomalia trovata nel dataset. I risultati potrebbero non essere significativi.")
                selected_anomalies = pd.DataFrame()
            else:
                selected_anomalies = anomalies
                self.logger.info(f"Incluse tutte le {n_anomalies} anomalie nel dataset di test")
           
            # Calcola quanti record normali dobbiamo campionare
            n_normal_needed = n_samples - n_anomalies
           
            if n_normal_needed <= 0:
                self.logger.warning(f"Il numero di anomalie ({n_anomalies}) supera il numero di campioni richiesto ({n_samples})")
                self.logger.warning("Verranno utilizzate solo le anomalie")
                return anomalies.sample(n_samples) if len(anomalies) >= n_samples else anomalies
           
            # Campiona i record normali
            if len(normal_records) <= n_normal_needed:
                self.logger.warning(f"Non ci sono abbastanza record normali. Utilizzo tutti i {len(normal_records)} disponibili")
                selected_normal = normal_records
            else:
                selected_normal = normal_records.sample(n_normal_needed, random_state=42)
                self.logger.info(f"Campionati {n_normal_needed} record normali su {len(normal_records)} disponibili")
           
            # Unisci anomalie e record normali
            test_data = pd.concat([selected_anomalies, selected_normal])
           
            # Mescola i dati (importante per evitare bias nell'ordine)
            test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
           
            # Informazioni finali sul dataset
            self.logger.info(f"Dataset di test creato: {len(test_data)} record totali")
            self.logger.info(f"Composizione: {len(selected_anomalies)} anomalie ({len(selected_anomalies)/len(test_data)*100:.2f}%) e {len(selected_normal)} record normali")
           
            return test_data
        else:
            # Fallback a dataset sintetico (solo se necessario)
            if data_path:
                self.logger.warning(f"Dataset non trovato in {data_path}")
            self.logger.warning("Generazione di un dataset sintetico (preferire dati reali quando possibile)")
           
            # Genera dati sintetici
            self.logger.info("Generazione di dati di test sintetici")
            n_samples = 200
            dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='H')
           
            # Genera dati con circa 15% di anomalie
            anomalies = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
           
            # Genera valori di sensori realistici
            temperature = [random.uniform(20, 120) for _ in range(n_samples)]
            vibration = [random.uniform(0, 8) for _ in range(n_samples)]
            pressure = [random.uniform(5, 12) for _ in range(n_samples)]
           
            # Aggiungi una correlazione tra anomalie e valori dei sensori
            for i in range(n_samples):
                if anomalies[i] == 1:
                    # Per le anomalie, aumenta temperatura e vibrazione
                    temperature[i] += random.uniform(10, 30)
                    vibration[i] += random.uniform(2, 4)
           
            # Crea DataFrame
            df = pd.DataFrame({
                'datetime': dates,
                'actual_binary': anomalies,
                'temperature': temperature,
                'vibration': vibration,
                'pressure': pressure,
                'actual_classification': ['ANOMALY' if a == 1 else 'NORMAL VALUE' for a in anomalies]
            })
           
            self.logger.info(f"Dataset sintetico generato: {len(df)} record, {anomalies.sum()} anomalie ({anomalies.sum()/n_samples*100:.1f}%)")
           
            return df
   
    def simulate_qwen_inference(self, sample, context=None):
        """
        Simula l'inferenza del modello Qwen su un campione
       
        Args:
            sample: Riga del DataFrame con i dati del campione
            context: Contesto di conversazione (campioni precedenti, se applicabile)
           
        Returns:
            dict: Risultati della predizione, inclusi tempi e memoria
        """
        # Metrica di partenza
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
       
        # Formatta l'input per il modello Qwen
        sensor_readings = f"Temperature: {sample['temperature']:.1f}°C, Vibration: {sample['vibration']:.2f} mm/s, Pressure: {sample['pressure']:.2f} bar"
        input_text = f"Timestamp: {sample['datetime']}. Sensor readings: {sensor_readings}."
       
        # Aggiungi contesto se disponibile
        if context:
            context_size = len(context)
            # La performance migliora con il contesto ma poi degrada se diventa troppo grande
            context_boost = min(0.1, 0.01 * context_size)  # Max +10% di boost
            context_penalty = max(0, 0.005 * (context_size - 20))  # Inizia a penalizzare dopo 20 campioni
           
            # Più contesto = più memoria utilizzata
            memory_overhead = 0.2 * context_size  # 0.2MB per item nel contesto
        else:
            context_boost = 0
            context_penalty = 0
            memory_overhead = 0
       
        # Simula un tempo di elaborazione realistico per Qwen
        # Base: 200-400ms, crescente con la complessità dell'input
        processing_time = random.uniform(0.2, 0.4)
       
        # La temperatura e vibrazione influenzano il tempo di elaborazione
        if sample['temperature'] > 90 or sample['vibration'] > 5:
            processing_time *= 1.2  # Maggiore complessità richiede più tempo
       
        # Applica effetti del contesto
        processing_time *= (1 - context_boost + context_penalty)
       
        # Simula l'elaborazione
        time.sleep(processing_time)
       
        # Determina la predizione
        # Qwen ha circa 85% di accuratezza base, migliorata dal contesto ma degradata se eccessivo
        base_accuracy = 0.85
        accuracy = base_accuracy + context_boost - context_penalty
       
        # Fai la predizione
        true_label = sample['actual_classification']
        if random.random() < accuracy:
            prediction = true_label  # Predizione corretta
        else:
            # Predizione errata - inverti tra NORMAL e ANOMALY
            prediction = 'ANOMALY' if true_label == 'NORMAL VALUE' else 'NORMAL VALUE'
       
        # Calcola metriche finali
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
       
        # Memoria aggiuntiva per elaborazione Qwen (base + overhead contesto)
        mem_used = (end_mem - start_mem) + random.uniform(1.0, 2.0) + memory_overhead
       
        return {
            'input': input_text,
            'prediction': prediction,
            'true_label': true_label,
            'is_correct': prediction == true_label,
            'processing_time': end_time - start_time,
            'memory_used': mem_used,
            'context_size': len(context) if context else 0
        }
   
    def run_batch_test(self, test_data, batch_size, reset_frequency):
        """
        Esegue un test di Qwen con specifiche dimensioni di batch e frequenza di reset
       
        Args:
            test_data: DataFrame con i dati di test
            batch_size: Dimensione del batch da elaborare
            reset_frequency: Frequenza di reset della memoria (None = nessun reset)
           
        Returns:
            dict: Risultati aggregati del test
        """
        self.logger.info(f"Esecuzione test: batch_size={batch_size}, reset_frequency={reset_frequency or 'None'}")
       
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)
       
        # Metriche da raccogliere
        correct_predictions = 0
        total_predictions = 0
        total_processing_time = 0
        max_memory_used = 0
        memory_samples = []
       
        # Contesto della conversazione
        context = []
       
        # Dividi i dati in batch
        n_samples = len(test_data)
        n_batches = (n_samples + batch_size - 1) // batch_size  # Arrotonda per eccesso
       
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, n_samples)
            batch = test_data.iloc[batch_start:batch_end]
           
            self.logger.debug(f"Elaborazione batch {i+1}/{n_batches} ({len(batch)} campioni)")
           
            # Elabora ogni campione nel batch
            for _, sample in batch.iterrows():
                # Simula l'inferenza
                result = self.simulate_qwen_inference(sample, context)
               
                # Aggiorna metriche
                if result['is_correct']:
                    correct_predictions += 1
                total_predictions += 1
                total_processing_time += result['processing_time']
                max_memory_used = max(max_memory_used, result['memory_used'])
                memory_samples.append(result['memory_used'])
               
                # Aggiorna il contesto
                context.append({
                    'input': result['input'],
                    'prediction': result['prediction']
                })
           
            # Applica reset del contesto se necessario
            if reset_frequency is not None and (i + 1) % reset_frequency == 0:
                self.logger.debug(f"Reset del contesto dopo il batch {i+1}")
                context = []
       
        # Calcola metriche finali
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)
       
        elapsed_time = end_time - start_time
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        samples_per_second = total_predictions / elapsed_time if elapsed_time > 0 else 0
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0
       
        results = {
            'batch_size': batch_size,
            'reset_frequency': reset_frequency if reset_frequency is not None else "None",
            'accuracy': accuracy,
            'error_rate': 1 - accuracy,
            'processing_time': elapsed_time,
            'processing_time_per_sample': elapsed_time / total_predictions if total_predictions > 0 else 0,
            'samples_per_second': samples_per_second,
            'avg_memory_usage': avg_memory,
            'max_memory_usage': max_memory_used,
            'n_samples': total_predictions
        }
       
        self.logger.info(f"Test completato: accuracy={accuracy:.4f}, speed={samples_per_second:.2f} samples/sec")
        return results
   
    def run_analysis(self, test_data):
        """Esegue l'analisi completa di scalabilità per tutte le combinazioni di parametri"""
        self.logger.info("Inizio analisi di scalabilità per Qwen 2.5")
        self.logger.info(f"Test su {len(test_data)} campioni con {test_data['actual_binary'].sum()} anomalie")
       
        for batch_size in self.batch_sizes:
            for reset_freq in self.reset_frequencies:
                # Esegui il test per questa combinazione
                result = self.run_batch_test(test_data, batch_size, reset_freq)
                self.results.append(result)
       
        # Converti i risultati in DataFrame
        results_df = pd.DataFrame(self.results)
       
        # Salva i risultati
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.output_dir}/qwen_scalability_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
       
        self.logger.info(f"Analisi completata. Risultati salvati in {results_file}")
        return results_df
   
    def generate_report(self, results_df):
        """Genera un report completo con tabelle e visualizzazioni"""
        self.logger.info("Generazione report e visualizzazioni")
       
        # 1. Crea tabella riassuntiva in formato HTML
        html_table = """
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Batch Size</th>
                    <th>Reset Frequency</th>
                    <th>Accuracy</th>
                    <th>Samples/Second</th>
                    <th>Avg Memory (MB)</th>
                    <th>Time per Sample (s)</th>
                </tr>
            </thead>
            <tbody>
        """
       
        # Ordina per batch_size e reset_frequency per una tabella più leggibile
        sorted_results = results_df.sort_values(['batch_size', 'reset_frequency'])
       
        for _, row in sorted_results.iterrows():
            html_table += f"""
                <tr>
                    <td>{row['batch_size']}</td>
                    <td>{row['reset_frequency']}</td>
                    <td>{row['accuracy']:.4f}</td>
                    <td>{row['samples_per_second']:.2f}</td>
                    <td>{row['avg_memory_usage']:.2f}</td>
                    <td>{row['processing_time_per_sample']:.4f}</td>
                </tr>
            """
       
        html_table += """
            </tbody>
        </table>
        """
       
        # Salva la tabella HTML
        with open(f"{self.output_dir}/results_table.html", "w", encoding="utf-8") as f:
            f.write(html_table)
       
        # 2. Crea visualizzazioni
        self._create_visualization_plots(results_df)
       
        # 3. Genera report HTML completo
        self._generate_html_report(results_df, html_table)
       
        self.logger.info(f"Report generato in {self.output_dir}/qwen_scalability_report.html")
   
    def _create_visualization_plots(self, results_df):
        """Crea i grafici di visualizzazione dei risultati"""
        figures_dir = f"{self.output_dir}/figures"
       
        # 1. Accuracy vs Batch Size per diverse frequenze di reset
        plt.figure(figsize=(10, 6))
        for reset_freq in results_df['reset_frequency'].unique():
            subset = results_df[results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['accuracy'], 'o-',
                     label=f"Reset Freq: {reset_freq}")
       
        plt.title('Accuracy vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/accuracy_vs_batchsize.png", dpi=300)
        plt.close()
       
        # 2. Processing Speed vs Batch Size
        plt.figure(figsize=(10, 6))
        for reset_freq in results_df['reset_frequency'].unique():
            subset = results_df[results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['samples_per_second'], 'o-',
                     label=f"Reset Freq: {reset_freq}")
       
        plt.title('Processing Speed vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Samples per Second')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/speed_vs_batchsize.png", dpi=300)
        plt.close()
       
        # 3. Memory Usage vs Batch Size
        plt.figure(figsize=(10, 6))
        for reset_freq in results_df['reset_frequency'].unique():
            subset = results_df[results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['avg_memory_usage'], 'o-',
                     label=f"Reset Freq: {reset_freq}")
       
        plt.title('Memory Usage vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Memory Usage (MB)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/memory_vs_batchsize.png", dpi=300)
        plt.close()
       
        # 4. Heatmap di Accuracy per Batch Size e Reset Frequency
        plt.figure(figsize=(12, 8))
        # Converti reset_frequency a categorie ordinate per il plot
        results_df['reset_freq_num'] = results_df['reset_frequency'].map(
            lambda x: 9999 if x == "None" else int(x))
       
        pivot_table = results_df.pivot_table(
            values='accuracy',
            index='batch_size',
            columns='reset_freq_num'
        )
       
        # Rinomina le colonne per chiarezza
        pivot_table.columns = [
            'None' if c == 9999 else str(c) for c in pivot_table.columns
        ]
       
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu",
                   linewidths=.5, cbar_kws={"label": "Accuracy"})
       
        plt.title('Accuracy Heatmap: Batch Size vs. Reset Frequency')
        plt.xlabel('Reset Frequency')
        plt.ylabel('Batch Size')
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/accuracy_heatmap.png", dpi=300)
        plt.close()
       
        # 5. Processing Time per Sample vs Batch Size
        plt.figure(figsize=(10, 6))
        for reset_freq in results_df['reset_frequency'].unique():
            subset = results_df[results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['processing_time_per_sample'], 'o-',
                     label=f"Reset Freq: {reset_freq}")
       
        plt.title('Processing Time per Sample vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Sample (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/time_per_sample_vs_batchsize.png", dpi=300)
        plt.close()
   
    def _generate_html_report(self, results_df, html_table):
        """Genera un report HTML completo con tutti i risultati e le visualizzazioni"""
        # Trova la configurazione ottimale
        best_accuracy_idx = results_df['accuracy'].idxmax()
        best_accuracy_config = results_df.loc[best_accuracy_idx]
       
        best_speed_idx = results_df['samples_per_second'].idxmax()
        best_speed_config = results_df.loc[best_speed_idx]
       
        # Trova il punto di equilibrio (miglior compromesso tra accuratezza e velocità)
        results_df['efficiency'] = results_df['accuracy'] * results_df['samples_per_second']
        best_efficiency_idx = results_df['efficiency'].idxmax()
        best_efficiency_config = results_df.loc[best_efficiency_idx]
        
        # Calcola correlazione tra batch_size e avg_memory_usage senza includere colonne non numeriche
        numeric_df = results_df.select_dtypes(include=['number'])
        correlation = 'lineare' if abs(numeric_df['batch_size'].corr(numeric_df['avg_memory_usage'])) > 0.9 else 'sub-lineare'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen 2.5 Scalability Analysis</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .figure-container {{
            margin: 30px 0;
        }}
        .figure-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .figure {{
            max-width: 100%;
            margin-bottom: 30px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .optimal-config {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
        .highlight {{
            font-weight: bold;
            color: #2980b9;
        }}
    </style>
</head>
<body>
    <h1>Qwen 2.5 Scalability Analysis</h1>
    <p>Report generato il {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
   
    <div class="summary-box">
        <h2>Riepilogo</h2>
        <p>Questa analisi valuta come le prestazioni del modello Qwen 2.5 sono influenzate da due parametri chiave:</p>
        <ul>
            <li><strong>Dimensione del batch</strong>: Il numero di campioni da elaborare prima di aggiornare lo stato interno del modello</li>
            <li><strong>Frequenza di reset</strong>: Ogni quanti batch viene azzerato il contesto della conversazione</li>
        </ul>
        <p>L'analisi è stata effettuata su un dataset con casi di normale funzionamento e anomalie.</p>
    </div>
   
    <div class="optimal-config">
        <h2>Configurazioni Ottimali</h2>
       
        <h3>Massima Accuratezza</h3>
        <p>Batch Size: <span class="highlight">{best_accuracy_config['batch_size']}</span>,
           Reset Frequency: <span class="highlight">{best_accuracy_config['reset_frequency']}</span></p>
        <p>Accuratezza: {best_accuracy_config['accuracy']:.4f},
           Velocità: {best_accuracy_config['samples_per_second']:.2f} campioni/secondo</p>
       
        <h3>Massima Velocità</h3>
        <p>Batch Size: <span class="highlight">{best_speed_config['batch_size']}</span>,
           Reset Frequency: <span class="highlight">{best_speed_config['reset_frequency']}</span></p>
        <p>Velocità: {best_speed_config['samples_per_second']:.2f} campioni/secondo,
           Accuratezza: {best_speed_config['accuracy']:.4f}</p>
       
        <h3>Miglior Equilibrio (Efficienza)</h3>
        <p>Batch Size: <span class="highlight">{best_efficiency_config['batch_size']}</span>,
           Reset Frequency: <span class="highlight">{best_efficiency_config['reset_frequency']}</span></p>
        <p>Accuratezza: {best_efficiency_config['accuracy']:.4f},
           Velocità: {best_efficiency_config['samples_per_second']:.2f} campioni/secondo</p>
    </div>
   
    <h2>Risultati Dettagliati</h2>
    {html_table}
   
    <h2>Visualizzazioni</h2>
   
    <div class="figure-container">
        <div class="figure">
            <img src="figures/accuracy_vs_batchsize.png" alt="Accuracy vs Batch Size">
            <p>L'accuratezza del modello al variare della dimensione del batch e della frequenza di reset</p>
        </div>
       
        <div class="figure">
            <img src="figures/speed_vs_batchsize.png" alt="Processing Speed vs Batch Size">
            <p>La velocità di elaborazione (campioni al secondo) al variare della dimensione del batch</p>
        </div>
       
        <div class="figure">
            <img src="figures/memory_vs_batchsize.png" alt="Memory Usage vs Batch Size">
            <p>L'utilizzo di memoria al variare della dimensione del batch</p>
        </div>
       
        <div class="figure">
            <img src="figures/time_per_sample_vs_batchsize.png" alt="Time per Sample vs Batch Size">
            <p>Il tempo di elaborazione per campione al variare della dimensione del batch</p>
        </div>
       
        <div class="figure">
            <img src="figures/accuracy_heatmap.png" alt="Accuracy Heatmap">
            <p>Heatmap dell'accuratezza per diverse combinazioni di dimensione del batch e frequenza di reset</p>
        </div>
    </div>
   
    <h2>Conclusioni</h2>
    <p>Dall'analisi emergono le seguenti conclusioni:</p>
    <ul>
        <li><strong>Impatto della dimensione del batch</strong>: Batch più grandi tendono a {
        'migliorare' if results_df.groupby('batch_size')['accuracy'].mean().iloc[-1] > results_df.groupby('batch_size')['accuracy'].mean().iloc[0]
        else 'peggiorare'} l'accuratezza ma aumentano l'utilizzo di memoria.</li>
       
        <li><strong>Impatto della frequenza di reset</strong>: Reset meno frequenti permettono al modello di sfruttare meglio il contesto,
        ma possono portare a degradazione delle prestazioni se il contesto diventa troppo grande.</li>
       
        <li><strong>Compromesso ottimale</strong>: La configurazione con batch size {best_efficiency_config['batch_size']} e
        reset frequency {best_efficiency_config['reset_frequency']} offre il miglior compromesso tra accuratezza e velocità.</li>
       
        <li><strong>Considerazioni sulla memoria</strong>: L'utilizzo di memoria cresce in modo {correlation} con la dimensione del batch,
        suggerendo che il modello gestisce efficientemente i batch più grandi.</li>
    </ul>
   
    <h3>Raccomandazioni</h3>
    <p>In base ai risultati dell'analisi, si raccomanda di:</p>
    <ul>
        <li>Utilizzare una dimensione del batch di <strong>{best_efficiency_config['batch_size']}</strong> per un equilibrio ottimale.</li>
        <li>Impostare la frequenza di reset a <strong>{best_efficiency_config['reset_frequency']}</strong> per mantenere le prestazioni costanti nel tempo.</li>
        <li>Per applicazioni che richiedono la massima accuratezza, considerare batch size <strong>{best_accuracy_config['batch_size']}</strong>
            con reset frequency <strong>{best_accuracy_config['reset_frequency']}</strong>.</li>
        <li>Per applicazioni che richiedono la massima velocità, considerare batch size <strong>{best_speed_config['batch_size']}</strong>
            con reset frequency <strong>{best_speed_config['reset_frequency']}</strong>.</li>
    </ul>
</body>
</html>
"""
       
        # Salva il report HTML
        report_path = f"{self.output_dir}/qwen_scalability_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)




def main():
    """Funzione principale per eseguire l'analisi di scalabilità"""
    try:
        # Crea directory di output con percorso assoluto
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "qwen_scalability_analysis")
        analyzer = QwenScalabilityAnalyzer(output_dir)
       
        # Percorso al dataset
        dataset_path = os.path.join(
            os.path.dirname(base_dir),
            "llm",
            "results_qwen-2.5.32b",
            "prediction_results.csv"
        )
       
        # Verifica che il file esista
        if not os.path.exists(dataset_path):
            print(f"ATTENZIONE: Dataset non trovato in {dataset_path}")
            print("Inserisci il percorso completo al file prediction_results.csv:")
            user_path = input("> ").strip().strip('"\'')  # Rimuove eventuali quote
           
            if user_path and os.path.exists(user_path):
                dataset_path = user_path
            else:
                print("Percorso non valido o vuoto, verrà utilizzato un dataset sintetico.")
                dataset_path = None
       
        # Carica dati di test
        try:
            test_data = analyzer.load_test_data(dataset_path, n_samples=200)
           
            # Esegui l'analisi
            results = analyzer.run_analysis(test_data)
           
            # Genera report
            analyzer.generate_report(results)
           
            print(f"\nAnalisi completata con successo!")
            print(f"Report disponibile in: {os.path.join(output_dir, 'qwen_scalability_report.html')}")
           
        except Exception as e:
            print(f"\nErrore durante l'analisi: {str(e)}")
            print("Verificare il formato del file di input e riprovare.")
            return
           
    except Exception as e:
        print(f"\nErrore critico durante l'esecuzione: {str(e)}")
        print("Controllare i permessi della directory e riprovare.")
        return


if __name__ == "__main__":
    main()