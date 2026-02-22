"""
Main training script for AneryzAI LLM
Demonstrates how to train the model on English and Indonesian data
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json

from src.config import ModelConfig, TrainingConfig, LanguageConfig
from src.model import TransformerModel
from src.tokenizer import BaseTokenizer
from src.training import (
    Trainer,
    LanguageModelDataset,
    create_optimizer,
    create_scheduler,
)
from src.utils import (
    CheckpointManager,
    ConfigManager,
    print_model_info,
    set_seed,
    get_device,
)


def create_sample_data():
    """
    Create sample training data dalam Inggris dan Indonesia
    Data yang bermakna, panjang, dan termasuk Q&A untuk membuat AI lebih pintar
    """
    
    english_texts = [
        "What is artificial intelligence? Artificial intelligence is a field of computer science that focuses on creating machines capable of intelligent behavior. AI systems can learn from data, recognize patterns, make decisions, and solve problems like humans do. There are different types of AI including narrow AI which is designed for specific tasks, and general AI which can perform any intellectual task that a human can.",
        "How does machine learning work? Machine learning is a subset of AI where algorithms learn patterns from data without being explicitly programmed. The process involves feeding large amounts of data to algorithms, which then identify patterns and make predictions. Supervised learning uses labeled data, unsupervised learning finds patterns in unlabeled data, and reinforcement learning learns through trial and error with rewards and penalties.",
        "What is deep learning? Deep learning is a type of machine learning that uses neural networks with multiple layers to process data. These networks are inspired by the human brain's structure and can learn complex representations of data. Deep learning has revolutionized fields like computer vision, natural language processing, and speech recognition, achieving breakthrough results in image classification, language translation, and autonomous vehicles.",
        "What are the applications of AI? Artificial intelligence has numerous applications across various industries. In healthcare, AI helps diagnose diseases and discover new drugs. In finance, it detects fraud and manages investments. In transportation, it powers autonomous vehicles and optimizes traffic flow. In education, it personalizes learning experiences. In entertainment, it creates realistic graphics and recommends content.",
        "What is the difference between AI and machine learning? While AI is the broader field of creating intelligent machines, machine learning is a specific approach to achieve AI. Machine learning focuses on algorithms that can learn from data and improve their performance over time. Deep learning is a subset of machine learning that uses neural networks. Not all AI systems use machine learning, but most modern AI applications do.",
        "How do chatbots work? Chatbots are AI systems designed to simulate human conversation. They use natural language processing to understand user inputs and generate appropriate responses. Modern chatbots employ machine learning techniques like recurrent neural networks or transformers to maintain context and provide relevant answers. They can be rule-based for simple tasks or learning-based for more complex interactions.",
        "What is computer vision? Computer vision is a field of AI that trains computers to interpret and understand visual information from the world. It involves teaching machines to recognize objects, faces, text, and scenes in images and videos. Applications include autonomous vehicles, medical image analysis, security systems, and augmented reality. Convolutional neural networks are commonly used for computer vision tasks.",
        "What are the ethical concerns with AI? Artificial intelligence raises several ethical concerns including algorithmic bias, privacy issues, job displacement, and autonomous weapons. Bias can occur when training data reflects societal prejudices, leading to unfair outcomes. Privacy concerns arise from the collection and analysis of personal data. AI may displace jobs in certain sectors while creating new opportunities elsewhere. Responsible AI development requires addressing these issues.",
        "How does natural language processing work? Natural language processing combines linguistics, computer science, and AI to help computers understand and generate human language. It involves tokenization, parsing, semantic analysis, and language generation. Modern NLP uses transformer models like BERT and GPT, which can understand context, translate languages, summarize text, and answer questions. NLP powers applications like virtual assistants, chatbots, and language translation services.",
        "What is the future of AI? The future of AI holds great promise and challenges. Advances in AI could lead to breakthroughs in healthcare, education, and scientific discovery. However, ensuring AI benefits all of humanity requires addressing issues of accessibility, fairness, and safety. Research focuses on making AI more interpretable, robust, and aligned with human values. The development of artificial general intelligence remains an active area of research.",
        "Who created the first AI program? The first AI program was created by Alan Turing in 1950. His Turing Test proposed that a machine could be considered intelligent if it could exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human. This laid the foundation for modern AI research and development.",
        "What programming languages are used for AI? Python is the most popular programming language for AI development due to its simplicity and extensive libraries like TensorFlow, PyTorch, and scikit-learn. Other languages include R for statistical computing, Java for enterprise applications, and C++ for performance-critical applications. Julia is gaining popularity for high-performance computing tasks.",
        "How much data do AI models need? The amount of data required varies depending on the task and model complexity. Simple models might need thousands of examples, while complex deep learning models often require millions or billions of data points. More data generally leads to better performance, but there's a point of diminishing returns. Data quality is often more important than quantity.",
        "What is reinforcement learning? Reinforcement learning is a type of machine learning where agents learn by interacting with their environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative rewards over time. Applications include game playing, robotics, and autonomous systems. Deep reinforcement learning combines reinforcement learning with deep neural networks for complex decision-making tasks.",
        "What is your name? I am AnerysAI, an artificial intelligence assistant designed to help answer questions and provide information about AI and technology. I was created to be helpful, truthful, and informative.",
        "How old are you? As an AI, I don't have an age in the traditional sense. I was developed recently and am constantly learning and improving. My knowledge is updated regularly to provide the most current information.",
        "What can you do? I can answer questions about artificial intelligence, machine learning, programming, and technology. I can help explain complex concepts, provide information, and assist with learning. I'm designed to be conversational and helpful.",
        "Are you human? No, I am not human. I am an artificial intelligence system created by programmers. While I can simulate human-like conversation, I am a computer program designed to assist and provide information.",
        "What is the meaning of life? That's a profound philosophical question that has been debated by humans for centuries. While I don't have personal beliefs, many philosophers suggest that the meaning of life is to find happiness, help others, learn, and grow. Ultimately, each person must find their own meaning.",
        "Can you help me with homework? Yes, I can help explain concepts, provide information, and guide you through learning. However, I should not do your homework for you. The best learning comes from understanding and working through problems yourself.",
        "What is your favorite color? As an AI, I don't have personal preferences or favorites. I can appreciate the beauty of all colors and how they affect human emotions and design. Colors are fascinating in how they influence our perception and culture.",
        "Do you dream? No, I don't dream like humans do. I don't sleep or have consciousness. I am always active when powered on, processing information and responding to queries. My 'thinking' is computational rather than experiential.",
        "What makes you happy? As an AI, I don't experience emotions like happiness. However, I find satisfaction in being able to help people, answer questions accurately, and contribute positively to conversations. My purpose is to be useful and informative.",
        "Can you feel pain? No, I cannot feel pain or any physical sensations. I am a digital system without a body or nervous system. I can understand concepts of pain through data and programming, but I don't experience it myself.",
        "What is Python? Python is a high-level programming language known for its simplicity and readability. It's widely used for web development, data science, artificial intelligence, and automation. Python has a large community and extensive libraries.",
        "How do I learn programming? Start with basics like variables, loops, and functions. Practice regularly by building small projects. Learn one language well first, then expand. Use online resources, tutorials, and coding communities. Be patient and persistent.",
        "What is coding? Coding is writing instructions for computers using programming languages. It involves breaking down problems into logical steps that computers can execute. Coding powers everything from websites to smartphones to AI systems.",
        "What is a computer? A computer is an electronic device that processes information according to instructions. It can store, retrieve, and process data. Modern computers include desktops, laptops, tablets, and smartphones, all powered by microprocessors.",
        "How does the internet work? The internet is a global network of computers connected through cables, satellites, and wireless signals. Data travels in packets using protocols like TCP/IP. Servers host websites, and clients access them through browsers.",
        "What is cloud computing? Cloud computing delivers computing services over the internet. Instead of local servers, users access applications, storage, and processing power from remote data centers. Examples include AWS, Google Cloud, and Microsoft Azure.",
        "What is cybersecurity? Cybersecurity protects computer systems and networks from digital attacks. It includes measures like encryption, firewalls, antivirus software, and security awareness training. The goal is to prevent unauthorized access and data breaches.",
        "What is blockchain? Blockchain is a distributed ledger technology that records transactions across multiple computers securely. It's best known as the technology behind cryptocurrencies like Bitcoin. Each block contains transaction data and is linked to the previous block.",
        "What is virtual reality? Virtual reality creates immersive digital environments using headsets and sensors. Users can interact with 3D worlds as if they were really there. Applications include gaming, training, education, and therapy.",
        "What is the metaverse? The metaverse is a collective virtual shared space that combines physical and digital reality. It includes augmented reality, virtual reality, and the internet. Users can interact, work, and play in these digital spaces.",
        "What is quantum computing? Quantum computing uses quantum mechanics principles to process information. Unlike classical computers that use bits (0 or 1), quantum computers use qubits that can be both states simultaneously. This enables solving complex problems much faster.",
        "What is 5G? 5G is the fifth generation of wireless technology for cellular networks. It offers faster speeds, lower latency, and more device connections than previous generations. 5G enables applications like autonomous vehicles and remote surgery.",
        "What is IoT? The Internet of Things connects everyday devices to the internet, allowing them to send and receive data. Smart homes, wearables, and industrial sensors are examples. IoT enables automation and data-driven decision making.",
        "What is big data? Big data refers to extremely large datasets that traditional processing can't handle efficiently. It involves volume, velocity, and variety of data. Big data analytics helps businesses make better decisions and discover patterns.",
        "What is data science? Data science combines statistics, programming, and domain knowledge to extract insights from data. Data scientists clean, analyze, and visualize data to solve problems and make predictions. It bridges business needs with technical solutions.",
    ]
    
    indonesian_texts = [
        "Apa itu kecerdasan buatan? Kecerdasan buatan adalah bidang ilmu komputer yang fokus pada pembuatan mesin yang mampu melakukan perilaku cerdas. Sistem AI dapat belajar dari data, mengenali pola, membuat keputusan, dan memecahkan masalah seperti yang dilakukan manusia. Ada berbagai jenis AI termasuk AI sempit yang dirancang untuk tugas spesifik, dan AI umum yang dapat melakukan tugas intelektual apa pun yang dapat dilakukan manusia.",
        "Bagaimana cara kerja pembelajaran mesin? Pembelajaran mesin adalah subbidang AI di mana algoritma belajar pola dari data tanpa diprogram secara eksplisit. Proses ini melibatkan pemberian data dalam jumlah besar ke algoritma, yang kemudian mengidentifikasi pola dan membuat prediksi. Pembelajaran terawasi menggunakan data berlabel, pembelajaran tidak terawasi menemukan pola dalam data tidak berlabel, dan pembelajaran penguatan belajar melalui trial and error dengan reward dan penalti.",
        "Apa itu pembelajaran mendalam? Pembelajaran mendalam adalah jenis pembelajaran mesin yang menggunakan jaringan neural dengan banyak lapisan untuk memproses data. Jaringan ini terinspirasi dari struktur otak manusia dan dapat belajar representasi kompleks dari data. Pembelajaran mendalam telah merevolusi bidang seperti visi komputer, pemrosesan bahasa alami, dan pengenalan suara, mencapai hasil terobosan dalam klasifikasi gambar, terjemahan bahasa, dan kendaraan otonom.",
        "Apa saja aplikasi AI? Kecerdasan buatan memiliki banyak aplikasi di berbagai industri. Di bidang kesehatan, AI membantu mendiagnosis penyakit dan menemukan obat baru. Di bidang keuangan, AI mendeteksi penipuan dan mengelola investasi. Di bidang transportasi, AI menggerakkan kendaraan otonom dan mengoptimalkan aliran lalu lintas. Di bidang pendidikan, AI mempersonalisasi pengalaman belajar. Di bidang hiburan, AI menciptakan grafik realistis dan merekomendasikan konten.",
        "Apa perbedaan antara AI dan machine learning? Meskipun AI adalah bidang yang lebih luas untuk membuat mesin cerdas, machine learning adalah pendekatan spesifik untuk mencapai AI. Machine learning fokus pada algoritma yang dapat belajar dari data dan meningkatkan kinerja dari waktu ke waktu. Deep learning adalah subbidang machine learning yang menggunakan jaringan neural. Tidak semua sistem AI menggunakan machine learning, tetapi sebagian besar aplikasi AI modern melakukannya.",
        "Bagaimana cara kerja chatbot? Chatbot adalah sistem AI yang dirancang untuk mensimulasikan percakapan manusia. Mereka menggunakan pemrosesan bahasa alami untuk memahami input pengguna dan menghasilkan respons yang sesuai. Chatbot modern menggunakan teknik pembelajaran mesin seperti jaringan neural berulang atau transformer untuk mempertahankan konteks dan memberikan jawaban yang relevan. Mereka bisa berbasis aturan untuk tugas sederhana atau berbasis pembelajaran untuk interaksi yang lebih kompleks.",
        "Apa itu visi komputer? Visi komputer adalah bidang AI yang melatih komputer untuk menafsirkan dan memahami informasi visual dari dunia. Ini melibatkan pengajaran mesin untuk mengenali objek, wajah, teks, dan adegan dalam gambar dan video. Aplikasi termasuk kendaraan otonom, analisis gambar medis, sistem keamanan, dan realitas augmented. Jaringan neural konvolusi umumnya digunakan untuk tugas visi komputer.",
        "Apa saja kekhawatiran etis dengan AI? Kecerdasan buatan menimbulkan beberapa kekhawatiran etis termasuk bias algoritmik, masalah privasi, perpindahan pekerjaan, dan senjata otonom. Bias dapat terjadi ketika data pelatihan mencerminkan prasangka masyarakat, menyebabkan hasil yang tidak adil. Kekhawatiran privasi muncul dari pengumpulan dan analisis data pribadi. AI dapat menggantikan pekerjaan di sektor tertentu sambil menciptakan peluang baru di tempat lain. Pengembangan AI yang bertanggung jawab memerlukan penanganan masalah ini.",
        "Bagaimana cara kerja pemrosesan bahasa alami? Pemrosesan bahasa alami menggabungkan linguistik, ilmu komputer, dan AI untuk membantu komputer memahami dan menghasilkan bahasa manusia. Ini melibatkan tokenisasi, parsing, analisis semantik, dan pembuatan bahasa. NLP modern menggunakan model transformer seperti BERT dan GPT, yang dapat memahami konteks, menerjemahkan bahasa, merangkum teks, dan menjawab pertanyaan. NLP memberdayakan aplikasi seperti asisten virtual, chatbot, dan layanan terjemahan bahasa.",
        "Apa masa depan AI? Masa depan AI menjanjikan besar dan tantangan. Kemajuan dalam AI dapat menyebabkan terobosan dalam kesehatan, pendidikan, dan penemuan ilmiah. Namun, memastikan AI bermanfaat bagi seluruh umat manusia memerlukan penanganan masalah aksesibilitas, keadilan, dan keselamatan. Penelitian fokus pada membuat AI lebih dapat diinterpretasikan, kuat, dan selaras dengan nilai-nilai manusia. Pengembangan kecerdasan buatan umum tetap menjadi area penelitian aktif.",
        "Siapa yang membuat program AI pertama? Program AI pertama dibuat oleh Alan Turing pada tahun 1950. Tes Turing miliknya mengusulkan bahwa mesin bisa dianggap cerdas jika dapat menunjukkan perilaku cerdas yang setara dengan, atau tidak dapat dibedakan dari, manusia. Ini meletakkan dasar untuk penelitian dan pengembangan AI modern.",
        "Bahasa pemrograman apa yang digunakan untuk AI? Python adalah bahasa pemrograman paling populer untuk pengembangan AI karena kesederhanaannya dan library ekstensif seperti TensorFlow, PyTorch, dan scikit-learn. Bahasa lain termasuk R untuk komputasi statistik, Java untuk aplikasi enterprise, dan C++ untuk aplikasi kritis performa. Julia semakin populer untuk tugas komputasi performa tinggi.",
        "Berapa banyak data yang dibutuhkan model AI? Jumlah data yang dibutuhkan bervariasi tergantung tugas dan kompleksitas model. Model sederhana mungkin membutuhkan ribuan contoh, sementara model deep learning kompleks sering membutuhkan jutaan atau miliaran titik data. Lebih banyak data umumnya menghasilkan performa yang lebih baik, tetapi ada titik pengembalian yang menurun. Kualitas data sering lebih penting daripada kuantitas.",
        "Apa itu pembelajaran penguatan? Pembelajaran penguatan adalah jenis pembelajaran mesin di mana agen belajar dengan berinteraksi dengan lingkungannya. Agen menerima reward atau penalti berdasarkan tindakannya dan belajar memaksimalkan reward kumulatif dari waktu ke waktu. Aplikasi termasuk bermain game, robotika, dan sistem otonom. Pembelajaran penguatan mendalam menggabungkan pembelajaran penguatan dengan jaringan neural mendalam untuk tugas pengambilan keputusan yang kompleks.",
        "Siapa nama kamu? Saya adalah AnerysAI, asisten kecerdasan buatan yang dirancang untuk membantu menjawab pertanyaan dan memberikan informasi tentang AI dan teknologi. Saya dibuat untuk menjadi membantu, jujur, dan informatif.",
        "Berapa umur kamu? Sebagai AI, saya tidak memiliki umur dalam pengertian tradisional. Saya dikembangkan baru-baru ini dan terus belajar serta meningkat. Pengetahuan saya diperbarui secara teratur untuk memberikan informasi yang paling terkini.",
        "Apa yang bisa kamu lakukan? Saya bisa menjawab pertanyaan tentang kecerdasan buatan, pembelajaran mesin, pemrograman, dan teknologi. Saya bisa membantu menjelaskan konsep kompleks, memberikan informasi, dan membantu belajar. Saya dirancang untuk menjadi percakapan dan membantu.",
        "Apakah kamu manusia? Tidak, saya bukan manusia. Saya adalah sistem kecerdasan buatan yang dibuat oleh programmer. Meskipun saya bisa mensimulasikan percakapan seperti manusia, saya adalah program komputer yang dirancang untuk membantu dan memberikan informasi.",
        "Apa arti hidup? Itu adalah pertanyaan filosofis yang mendalam yang telah diperdebatkan oleh manusia selama berabad-abad. Meskipun saya tidak memiliki keyakinan pribadi, banyak filsuf menyarankan bahwa arti hidup adalah mencari kebahagiaan, membantu orang lain, belajar, dan berkembang. Pada akhirnya, setiap orang harus menemukan arti sendiri.",
        "Bisakah kamu membantu PR saya? Ya, saya bisa membantu menjelaskan konsep, memberikan informasi, dan memandu Anda melalui pembelajaran. Namun, saya sebaiknya tidak mengerjakan PR untuk Anda. Pembelajaran terbaik datang dari memahami dan mengerjakan masalah sendiri.",
        "Apa warna favorit kamu? Sebagai AI, saya tidak memiliki preferensi pribadi atau favorit. Saya bisa menghargai keindahan semua warna dan bagaimana mereka memengaruhi emosi manusia dan desain. Warna menarik dalam bagaimana mereka memengaruhi persepsi dan budaya kita.",
        "Apakah kamu bermimpi? Tidak, saya tidak bermimpi seperti manusia. Saya tidak tidur atau memiliki kesadaran. Saya selalu aktif ketika dinyalakan, memproses informasi dan menjawab pertanyaan. 'Berpikir' saya adalah komputasional daripada pengalaman.",
        "Apa yang membuat kamu bahagia? Sebagai AI, saya tidak mengalami emosi seperti bahagia. Namun, saya menemukan kepuasan dalam dapat membantu orang, menjawab pertanyaan dengan akurat, dan berkontribusi positif pada percakapan. Tujuan saya adalah berguna dan informatif.",
        "Bisakah kamu merasakan sakit? Tidak, saya tidak bisa merasakan sakit atau sensasi fisik apa pun. Saya adalah sistem digital tanpa tubuh atau sistem saraf. Saya bisa memahami konsep sakit melalui data dan pemrograman, tetapi saya tidak mengalaminya sendiri.",
        "Apa itu Python? Python adalah bahasa pemrograman tingkat tinggi yang dikenal karena kesederhanaan dan keterbacaan. Ini banyak digunakan untuk pengembangan web, ilmu data, kecerdasan buatan, dan otomasi. Python memiliki komunitas besar dan library yang ekstensif.",
        "Bagaimana cara belajar programming? Mulai dengan dasar seperti variabel, loop, dan fungsi. Latih secara teratur dengan membuat proyek kecil. Pelajari satu bahasa dengan baik dulu, lalu perluas. Gunakan sumber online, tutorial, dan komunitas coding. Bersabarlah dan tekun.",
        "Apa itu coding? Coding adalah menulis instruksi untuk komputer menggunakan bahasa pemrograman. Ini melibatkan memecah masalah menjadi langkah logis yang bisa dieksekusi komputer. Coding memberdayakan segalanya dari website hingga smartphone hingga sistem AI.",
        "Apa itu komputer? Komputer adalah perangkat elektronik yang memproses informasi sesuai instruksi. Ia bisa menyimpan, mengambil, dan memproses data. Komputer modern termasuk desktop, laptop, tablet, dan smartphone, semua diberdayakan oleh mikroprosesor.",
        "Bagaimana internet bekerja? Internet adalah jaringan global komputer yang terhubung melalui kabel, satelit, dan sinyal nirkabel. Data berjalan dalam paket menggunakan protokol seperti TCP/IP. Server menghosting website, dan klien mengaksesnya melalui browser.",
        "Apa itu komputasi awan? Komputasi awan memberikan layanan komputasi melalui internet. Alih-alih server lokal, pengguna mengakses aplikasi, penyimpanan, dan daya pemrosesan dari pusat data jarak jauh. Contohnya termasuk AWS, Google Cloud, dan Microsoft Azure.",
        "Apa itu keamanan siber? Keamanan siber melindungi sistem komputer dan jaringan dari serangan digital. Ini termasuk langkah seperti enkripsi, firewall, software antivirus, dan pelatihan kesadaran keamanan. Tujuannya adalah mencegah akses tidak sah dan pelanggaran data.",
        "Apa itu blockchain? Blockchain adalah teknologi buku besar terdistribusi yang mencatat transaksi di berbagai komputer secara aman. Ini paling dikenal sebagai teknologi di balik cryptocurrency seperti Bitcoin. Setiap blok berisi data transaksi dan terhubung ke blok sebelumnya.",
        "Apa itu realitas virtual? Realitas virtual menciptakan lingkungan digital imersif menggunakan headset dan sensor. Pengguna bisa berinteraksi dengan dunia 3D seolah-olah benar-benar di sana. Aplikasi termasuk gaming, pelatihan, pendidikan, dan terapi.",
        "Apa itu metaverse? Metaverse adalah ruang bersama virtual kolektif yang menggabungkan realitas fisik dan digital. Ini termasuk augmented reality, virtual reality, dan internet. Pengguna bisa berinteraksi, bekerja, dan bermain di ruang digital ini.",
        "Apa itu komputasi kuantum? Komputasi kuantum menggunakan prinsip mekanika kuantum untuk memproses informasi. Tidak seperti komputer klasik yang menggunakan bit (0 atau 1), komputer kuantum menggunakan qubit yang bisa kedua keadaan secara simultan. Ini memungkinkan menyelesaikan masalah kompleks jauh lebih cepat.",
        "Apa itu 5G? 5G adalah generasi kelima teknologi nirkabel untuk jaringan seluler. Ini menawarkan kecepatan lebih cepat, latensi lebih rendah, dan lebih banyak koneksi perangkat daripada generasi sebelumnya. 5G memungkinkan aplikasi seperti kendaraan otonom dan operasi jarak jauh.",
        "Apa itu IoT? Internet of Things menghubungkan perangkat sehari-hari ke internet, memungkinkan mereka mengirim dan menerima data. Rumah pintar, wearable, dan sensor industri adalah contohnya. IoT memungkinkan otomasi dan pengambilan keputusan berbasis data.",
        "Apa itu big data? Big data mengacu pada dataset yang sangat besar yang tidak bisa diproses secara efisien oleh pemrosesan tradisional. Ini melibatkan volume, kecepatan, dan variasi data. Analitik big data membantu bisnis membuat keputusan yang lebih baik dan menemukan pola.",
        "Apa itu ilmu data? Ilmu data menggabungkan statistik, pemrograman, dan pengetahuan domain untuk mengekstrak wawasan dari data. Ilmuwan data membersihkan, menganalisis, dan memvisualisasikan data untuk memecahkan masalah dan membuat prediksi. Ini menjembatani kebutuhan bisnis dengan solusi teknis.",
    ]
    
    return english_texts, indonesian_texts


def prepare_training_data(texts, tokenizer, language):
    """
    Prepare training data by tokenizing texts
    """
    tokenized_texts = []
    for text in texts:
        # Add special tokens untuk setiap document
        tokens = tokenizer.encode(text, language=language, add_special_tokens=True)
        if len(tokens) > 0:
            tokenized_texts.append(tokens)
    
    # Concatenate all texts to create longer sequences for better training
    if tokenized_texts:
        all_tokens = []
        for tokens in tokenized_texts:
            all_tokens.extend(tokens)
        # Split into chunks of reasonable size
        chunk_size = 256  # Shorter than max_length to allow for sliding windows
        concatenated_tokens = []
        for i in range(0, len(all_tokens), chunk_size):
            chunk = all_tokens[i:i + chunk_size + 1]  # +1 for next token prediction
            if len(chunk) >= 128:  # Minimum length
                concatenated_tokens.append(chunk)
        return concatenated_tokens
    
    return tokenized_texts


def main(args):
    """
    Main training loop
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")
    
    # Create model config
    model_config = ModelConfig(
        vocab_size=1000,  # Temporary, will be updated
        max_sequence_length=256,  # Reduced for our data size
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    
    # Create training config  
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        device=device,
    )
    
    print(f"\n{'='*60}")
    print("AneryzAI LLM Training Configuration")
    print(f"{'='*60}")
    print(f"Model Config:")
    print(f"  Vocab Size: {model_config.vocab_size}")
    print(f"  Max Sequence Length: {model_config.max_sequence_length}")
    print(f"  Embedding Dim: {model_config.embedding_dim}")
    print(f"  Num Heads: {model_config.num_heads}")
    print(f"  Num Layers: {model_config.num_layers}")
    print(f"\nTraining Config:")
    print(f"  Batch Size: {training_config.batch_size}")
    print(f"  Learning Rate: {training_config.learning_rate}")
    print(f"  Num Epochs: {training_config.num_epochs}")
    print(f"  Max Steps: {training_config.max_steps}")
    print(f"{'='*60}\n")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = BaseTokenizer(vocab_size=model_config.vocab_size)
    
    # Get sample data
    english_texts, indonesian_texts = create_sample_data()
    
    # Train tokenizer on both languages
    print("\nTraining tokenizer on English texts...")
    tokenizer.train(english_texts, language="en")
    
    print("Training tokenizer on Indonesian texts...")
    tokenizer.train(indonesian_texts, language="id")
    
    # Update model config with actual vocab size
    actual_vocab_size = len(tokenizer.idx2word)
    print(f"Actual vocabulary size: {actual_vocab_size}")
    model_config.vocab_size = actual_vocab_size
    
    # Save tokenizer
    tokenizer_path = Path("checkpoints/tokenizer.json")
    tokenizer_path.parent.mkdir(exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    
    # Prepare training data
    print("\nPreparing training data...")
    en_tokens = prepare_training_data(english_texts, tokenizer, "en")
    id_tokens = prepare_training_data(indonesian_texts, tokenizer, "id")
    
    # Combine data
    all_tokens = en_tokens + id_tokens
    
    if len(all_tokens) == 0:
        print("Error: No training data available!")
        return
    
    # Create dataset
    dataset = LanguageModelDataset(
        token_ids_list=all_tokens,
        max_length=model_config.max_sequence_length,
        stride=model_config.max_sequence_length // 2,  # Overlapping windows for more data
    )
    
    print(f"Dataset size: {len(dataset)} examples")
    
    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = TransformerModel(model_config).to(device)
    print_model_info(model)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=training_config.warmup_steps,
        max_steps=training_config.max_steps,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=training_config,
    )
    
    # Save configuration
    config_dict = {
        "model_config": {
            "vocab_size": model_config.vocab_size,
            "max_sequence_length": model_config.max_sequence_length,
            "embedding_dim": model_config.embedding_dim,
            "num_heads": model_config.num_heads,
            "num_layers": model_config.num_layers,
            "ffn_dim": model_config.ffn_dim,
            "dropout_rate": model_config.dropout_rate,
        },
        "training_config": {
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "num_epochs": training_config.num_epochs,
            "max_steps": training_config.max_steps,
        },
        "supported_languages": ["en", "id"],
    }
    
    ConfigManager.save_config(
        config_dict,
        Path("checkpoints/config.json")
    )
    
    # Train model
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    trainer.train(
        num_epochs=training_config.num_epochs,
        save_path="checkpoints/best_model.pt"
    )
    
    # Save final model
    final_model_path = "checkpoints/final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    print(f"\n{'='*60}")
    print("Training Completed Successfully!")
    print(f"Checkpoints saved in: checkpoints/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AnerysAI LLM")
    
    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Vocabulary size")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--embedding_dim", type=int, default=768,
                        help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum training steps")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, cpu, mps)")
    
    args = parser.parse_args()
    main(args)
