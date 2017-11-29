#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <chrono>

#include <primitiv/primitiv.h>
#include <primitiv/primitiv_cuda.h>

using primitiv::optimizers::SGD;
using primitiv::initializers::Constant;
using primitiv::initializers::Uniform;
using primitiv::initializers::XavierUniform;
namespace F = primitiv::operators;

using namespace primitiv;
using namespace std;
using namespace std::chrono;

const string SRC_TRAIN = "../data/small_parallel_enja/train.en";
const string TRG_TRAIN = "../data/small_parallel_enja/train.ja";
const string SRC_TEST = "../data/small_parallel_enja/test.en";
const string TRG_TEST = "../data/small_parallel_enja/test.ja";
const int MAX_EPOCH = 30;

template<typename Var>
class LSTM : public primitiv::Model {
	primitiv::Parameter pwxh_, pwhh_, pbh_;
	Var wxh_, whh_, bh_, h_, c_;

	public:
	LSTM() {
		add_parameter("wxh", pwxh_);
		add_parameter("whh", pwhh_);
		add_parameter("bh", pbh_);
	}

	// Initializes the model.
	void init(unsigned in_size, unsigned out_size) {
		pwxh_.init({4 * out_size, in_size}, Uniform(-0.8, 0.8));
		pwhh_.init({4 * out_size, out_size}, Uniform(-0.8, 0.8));
		pbh_.init({4 * out_size}, Uniform(-0.8, 0.8));
	}

	// Initializes internal values.
	void reset(const Var &init_c = Var(), const Var &init_h = Var()) {
		const unsigned out_size = pwhh_.shape()[1];
		wxh_ = F::parameter<Var>(pwxh_);
		whh_ = F::parameter<Var>(pwhh_);
		bh_ = F::parameter<Var>(pbh_);
		c_ = init_c.valid() ? init_c : F::zeros<Var>({out_size});
		h_ = init_h.valid() ? init_h : F::zeros<Var>({out_size});
	}

	// One step forwarding.
	Var forward(const Var &x) {
		namespace F = primitiv::operators;
		const unsigned out_size = pwhh_.shape()[1];
		const auto u = F::matmul(wxh_, x) + F::matmul(whh_, h_) + bh_;
		const auto i = F::sigmoid(F::slice(u, 0, 0, out_size));
		const auto f = F::sigmoid(F::slice(u, 0, out_size, 2 * out_size));
		const auto o = F::sigmoid(F::slice(u, 0, 2 * out_size, 3 * out_size));
		const auto j = F::tanh(F::slice(u, 0, 3 * out_size, 4 * out_size));
		c_ = i * j + f * c_;
		h_ = o * F::tanh(c_);
		return h_;
	}

	// Retrieves current states.
	Var get_c() const { return c_; }
	Var get_h() const { return h_; }
};

template <typename Var>
class EncoderDecoder : public Model {
	Parameter psrc_lookup, ptrg_lookup, pw;
	LSTM<Var> src_lstm, trg_lstm;
	Var src_lookup, trg_lookup, w;

public:
	EncoderDecoder() {
		add_parameter("psrc_lookup", psrc_lookup);
		add_parameter("ptrg_lookup", ptrg_lookup);
		add_parameter("pw", pw);
		add_submodel("src_lstm", src_lstm);
		add_submodel("trg_lstm", trg_lstm);
	}

	void init(unsigned src_vocab_size, unsigned trg_vocab_size,
		unsigned embed_size, unsigned hidden_size) {
			psrc_lookup.init({embed_size, src_vocab_size}, XavierUniform());
			ptrg_lookup.init({embed_size, trg_vocab_size}, XavierUniform());
			pw.init({trg_vocab_size, hidden_size}, XavierUniform());

			src_lstm.init(embed_size, hidden_size);
			trg_lstm.init(embed_size, hidden_size);
	}

	void encode(const vector<vector<unsigned>> src) {
		src_lookup = F::parameter<Var>(psrc_lookup);
		src_lstm.reset();
		for(auto word : src) {
			Var x = F::pick(src_lookup, word, 1);
			src_lstm.forward(x);
		}

		trg_lookup = F::parameter<Var>(ptrg_lookup);
		w = F::parameter<Var>(pw);
		trg_lstm.reset();
	}

	Var decode_step(const vector<unsigned> trg_word) {
		Var x = F::pick(trg_lookup, trg_word, 1);
		Var h = trg_lstm.forward(x);
		return F::matmul(w, h);
	}

	Var loss(const vector<vector<unsigned>> trg) {
		vector<Var> losses;
		for(unsigned i = 0; i < trg.size() - 1; i++) {
			Var y = decode_step(trg[i]);
			losses.emplace_back(
					F::softmax_cross_entropy(y, trg[i+1], 0));
		}
		return F::batch::mean(F::sum(losses));
	}
};


inline unordered_map<string, unsigned> make_vocab(
		const string &path, unsigned size) {
	if (size < 3) {
		cerr << "Vocab size should be <= 3" << endl;
		exit(1);
	}

	ifstream ifs(path);
	if (!ifs.is_open()) {
		cerr << "File could not be opend: " << path << endl;
		exit(1);
	}

	unordered_map<string, unsigned> freq;
	string line, word;
	while (getline(ifs, line)) {
		stringstream ss(line);
		while (getline(ss, word, ' ')) ++freq[word];
	}

	using freq_t = pair<string, unsigned>;
	auto cmp = [](const freq_t &a, const freq_t &b) {
		return a.second < b.second;
	};
	priority_queue<freq_t, vector<freq_t>, decltype(cmp)> q(cmp);
	for (const auto &x : freq) q.push(x);

	unordered_map<string, unsigned> vocab;
	vocab.emplace("<unk>", 0);
	vocab.emplace("<bos>", 1);
	vocab.emplace("<eos>", 2);

	size = min<unsigned>(size, q.size());
	for (unsigned i = 3; i < size; ++i) {
		vocab.emplace(q.top().first, i);
		q.pop();
	}
	return vocab;
}

inline unsigned count_labels(const vector<vector<unsigned>> &corpus) {
	unsigned ret = 0;
	for (const auto &sent : corpus) ret += sent.size() - 1;
	return ret;
}

inline vector<vector<unsigned>> make_batch(
    const vector<vector<unsigned>> &corpus,
    const vector<unsigned> &sent_ids,
    const unordered_map<string, unsigned> &vocab) {
  const unsigned batch_size = sent_ids.size();
  const unsigned eos_id = vocab.at("<eos>");
  unsigned max_len = 0;
  for (const unsigned sid : sent_ids) {
    max_len = max<unsigned>(max_len, corpus[sid].size());
  }
  vector<vector<unsigned>> batch(
      max_len, vector<unsigned>(batch_size, eos_id));
  for (unsigned i = 0; i < batch_size; ++i) {
    const auto &sent = corpus[sent_ids[i]];
    for (unsigned j = 0; j < sent.size(); ++j) {
      batch[j][i] = sent[j];
    }
  }
  return batch;
}

inline vector<vector<unsigned>> load_corpus(
    const string &path,
    const unordered_map<string, unsigned> &vocab) {
  ifstream ifs(path);
	if (!ifs.is_open()) {
		cerr << "File could not be opened: " << path << endl;
		exit(1);
	}

  const unsigned unk_id = vocab.at("<unk>");
  vector<vector<unsigned>> corpus;
  string line, word;
  while (getline(ifs, line)) {
		string converted = "<bos> " + line + " <eos>";
		stringstream ss(converted);
		vector<unsigned> sent;
		string word;
		while (getline(ss, word, ' ')) {
			const auto it = vocab.find(word);
			if ( it != vocab.end() ) sent.emplace_back(it->second);
			else sent.emplace_back(unk_id);
		}
		corpus.emplace_back(sent);
	}
  return corpus;
}

int main(int argc, char** argv) {

	auto start = system_clock::now();

	if (argc != 7) {
		cerr << "Usage: " << argv[0];
		cerr << " gpu_id src_vocab trg_vocab embed_size";
		cerr << " hidden_size minibatch_size" << endl;
		exit(1);
	}

	const int gpu_device = atoi(argv[1]);
	const int src_vocab_size = atoi(argv[2]);
	const int trg_vocab_size = atoi(argv[3]);
	const int embed_size = atoi(argv[4]);
	const int hidden_size = atoi(argv[5]);
	const int batchsize = atoi(argv[6]);

	static Device *dev;
	if (gpu_device >= 0)
		dev = new devices::CUDA(gpu_device);
	else
		dev = new devices::Naive();
	Device::set_default(*dev);

	Graph g;
	Graph::set_default(g);

	EncoderDecoder<Node> encdec;
	encdec.init(src_vocab_size, trg_vocab_size, embed_size, hidden_size);

	SGD optimizer(0.7);
	optimizer.add_model(encdec);

	auto src_vocab = make_vocab(SRC_TRAIN, src_vocab_size);
	auto trg_vocab = make_vocab(TRG_TRAIN, trg_vocab_size);

	const auto src_train = load_corpus(SRC_TRAIN, src_vocab);
	const auto trg_train = load_corpus(TRG_TRAIN, trg_vocab);
	const auto src_test = load_corpus(SRC_TEST, src_vocab);
	const auto trg_test = load_corpus(TRG_TEST, trg_vocab);

	unsigned num_train_sents = trg_train.size();
	unsigned num_test_sents = trg_test.size();
	unsigned num_train_labels = count_labels(trg_train);
	unsigned num_test_labels = count_labels(trg_test);

	vector<unsigned> train_ids(num_train_sents);
	iota(begin(train_ids), end(train_ids), 0);
	vector<unsigned> test_ids(num_test_sents);
	iota(begin(test_ids), end(test_ids), 0);

	random_device rd;
	mt19937 rng(rd());

	{
		duration<float> fs = system_clock::now() - start;
		float startup_time = duration_cast<milliseconds>(fs).count();
		cout << "startup time=" << startup_time / 1000. << endl;
	}

	for (unsigned epoch = 0; epoch < MAX_EPOCH; epoch++) {
		start = system_clock::now();

		if (epoch > 5) {
			float new_scale = 0.5 * optimizer.get_learning_rate_scaling();
			optimizer.set_learning_rate_scaling(new_scale);
		}

		shuffle(begin(train_ids), end(train_ids), rng);
		for (unsigned ofs = 0; ofs < num_train_sents; ofs += batchsize) {
			float pos = ofs - num_train_sents / 2;
			if (epoch > 5 && 0 <= pos && pos < batchsize) {
				float new_scale = 0.5 * optimizer.get_learning_rate_scaling();
				optimizer.set_learning_rate_scaling(new_scale);
			}

			const vector<unsigned> batch_ids(
					begin(train_ids)+ofs, begin(train_ids) + min<unsigned>(
						ofs + batchsize, num_train_sents));
			const auto src_batch = make_batch(src_train, batch_ids, src_vocab);
			const auto trg_batch = make_batch(trg_train, batch_ids, trg_vocab);

			g.clear();
			encdec.encode(src_batch);
			const auto loss = encdec.loss(trg_batch);

			optimizer.reset_gradients();
			loss.backward();
			optimizer.update();
		}

		duration<float> fs = system_clock::now() - start;
		float train_time = duration_cast<milliseconds>(fs).count() / 1000.;

		float test_loss = 0;
		for (unsigned ofs = 0; ofs < num_test_sents; ofs += batchsize) {
			const vector<unsigned> batch_ids(
					begin(test_ids)+ofs, begin(test_ids) + min<unsigned>(
						ofs + batchsize, num_test_sents));
			const auto src_batch = make_batch(src_test, batch_ids, src_vocab);
			const auto trg_batch = make_batch(trg_test, batch_ids, trg_vocab);

			g.clear();
			encdec.encode(src_batch);
			const auto loss = encdec.loss(trg_batch);
			test_loss += loss.to_float() * batch_ids.size();
		}
		float test_ppl = exp(test_loss / num_test_labels);

		cout << "epoch=" << epoch + 1 << ", ";
		cout << "time=" << train_time << ", ";
		cout << "ppl=" << test_ppl << ", ";
		cout << "word_per_sec=" << num_train_labels / train_time << endl;
	}
}
