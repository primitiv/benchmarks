#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include <primitiv/primitiv.h>
#include <primitiv/primitiv_cuda.h>

using primitiv::initializers::Constant;
using primitiv::initializers::XavierUniform;
using primitiv::optimizers::Adam;
namespace F = primitiv::operators;
using namespace primitiv;
using namespace std;

const string TRAIN_FILE = "../data/text/ptb.train.txt";
const string TEST_FILE = "../data/text/ptb.test.txt";
const int MAX_EPOCH = 100;

template <typename Var>
class LSTM : public Model {
	unsigned out_size;
	Parameter pwxh, pwhh, pbh;
	Var wxh, whh, bh, h, c;

public:
	LSTM(unsigned in_size_, unsigned out_size_)
			: out_size(out_size_)
			, pwxh({4 * out_size_, in_size_}, XavierUniform())
			, pwhh({4 * out_size_, out_size_}, XavierUniform())
			, pbh({4 * out_size_}, Constant(0)){
		add_parameter("pwxh", pwxh);
		add_parameter("pwhh", pwhh);
		add_parameter("pbh", pbh);
	}

	void init() {
		wxh = F::parameter<Var>(pwxh);
		whh = F::parameter<Var>(pwhh);
		bh = F::parameter<Var>(pbh);
		h = c = F::zeros<Var>({out_size});
	}

	Var forward(const Var &x) {
		const Var u = F::matmul(wxh, x) + F::matmul(whh, h) + bh;
		const Var i = F::sigmoid(F::slice(u, 0, 0, out_size));
		const Var f = F::sigmoid(F::slice(u, 0, out_size, 2 * out_size));
		const Var o = F::sigmoid(F::slice(u, 0, 2 * out_size, 3 * out_size));
		const Var j = F::tanh(F::slice(u, 0, 3 * out_size, 4 * out_size));
		c = i * j + f * c;
		h = o * F::tanh(c);
		return h;
	}
};

template <typename Var>
class RNNLM : public Model {
	Parameter plookup;
	LSTM<Var> lstm;
	Parameter pwhy, pby;
	Var lookup, why, by;

public:
	RNNLM(unsigned vocab_size, unsigned embed_size, unsigned hidden_size)
			: plookup({embed_size, vocab_size}, XavierUniform())
			, lstm(embed_size, hidden_size)
			, pwhy({vocab_size, hidden_size}, XavierUniform())
			, pby({vocab_size}, Constant(0)) {
		add_parameter("plookup", plookup);
		add_submodel("lstm", lstm);
		add_parameter("pwhy", pwhy);
		add_parameter("pby", pby);
	}

	Var forward(const vector<unsigned> &input) {
		Var x = F::pick(lookup, input, 1);
		Var h = lstm.forward(x);
		return F::matmul(why, h) + by;
	}

	Var loss(const vector<vector<unsigned>> &inputs) {
		lookup = F::parameter<Var>(plookup);
		why = F::parameter<Var>(pwhy);
		by = F::parameter<Var>(pby);
		lstm.init();

		vector<Var> losses;
		for (unsigned i = 0; i < inputs.size()-1; i++) {
			const auto output = forward(inputs[i]);
			losses.emplace_back(
					F::softmax_cross_entropy(output, inputs[i + 1], 0));
		}
		return F::batch::mean(F::sum(losses));
	}
};


unordered_map<string, unsigned> make_vocab(
		const string &filename) {
	ifstream ifs(filename);
	if (!ifs.is_open()) {
		cerr << "File could not be opened: " << filename << endl;
		exit(1);
	}
	unordered_map<string, unsigned> vocab;
	string line, word;
	while (getline(ifs, line)) {
		line = line + " <s>";
		stringstream ss(line);
		while (getline(ss, word, ' ')) {
			if (vocab.find(word) != vocab.end())
				continue;
			const unsigned id = vocab.size();
			vocab.emplace(make_pair(word, id));
		}
	}
	return vocab;
}

vector<vector<unsigned>> load_corpus(
		const string &filename,
		const unordered_map<string, unsigned> &vocab) {
	ifstream ifs(filename);
	if (!ifs.is_open()) {
		cerr << "File could not be opened: " << filename << endl;
		exit(1);
	}
	vector<vector<unsigned>> corpus;
	string line, word;
	while (getline(ifs, line)) {
		line = line + " <s>";
		stringstream ss(line);
		vector<unsigned> sentence;
		while (getline(ss, word, ' ')) {
			sentence.emplace_back(vocab.at(word));
		}
		corpus.emplace_back(move(sentence));
	}
	return corpus;
}

unsigned count_labels(const vector<vector<unsigned>> &corpus) {
	unsigned ret = 0;
	for (const auto &sent : corpus) ret += sent.size() - 1;
	return ret;
}

vector<vector<unsigned>> make_batch(
		const vector<vector<unsigned>> &corpus,
		const vector<unsigned> &sent_ids,
		unsigned eos_id) {

	const unsigned batch_size = sent_ids.size();
	unsigned max_len = 0;
	for (const unsigned sid : sent_ids)
		max_len = max<unsigned>(max_len, corpus[sid].size());

	vector<vector<unsigned>> batch(max_len,
			vector<unsigned>(batch_size, eos_id));
	for (unsigned i = 0; i < batch_size; i++) {
		const auto &sent = corpus[sent_ids[i]];
		for (unsigned j = 0; j < sent.size(); j++) {
			batch[j][i] = sent[j];
		}
	}
	return batch;
}

int main(int argc, char** argv) {

	const int gpu_device = 1;
	const int embed = 64;
	const int hidden = 128;
	const int minibatch = 64;

	auto vocab = make_vocab(TRAIN_FILE);
	unsigned eos_id = vocab["<s>"];

	const auto train_corpus = load_corpus(TRAIN_FILE, vocab);
	const auto test_corpus = load_corpus(TEST_FILE, vocab);
	const unsigned num_train_sents = train_corpus.size();
	const unsigned num_test_sents = test_corpus.size();
	const unsigned num_train_labels = count_labels(train_corpus);
	const unsigned num_test_labels = count_labels(test_corpus);

	devices::CUDA dev(gpu_device);
	Device::set_default(dev);

	Graph g;
	Graph::set_default(g);

	RNNLM<Node> rnnlm(vocab.size(), embed, hidden);

	Adam optimizer;
	optimizer.add_model(rnnlm);

	random_device rd;
	mt19937 rng(rd());

	vector<unsigned> train_ids(num_train_sents);
	vector<unsigned> test_ids(num_test_sents);
	iota(begin(train_ids), end(train_ids), 0);
	iota(begin(test_ids), end(test_ids), 0);

	for (unsigned epoch = 0; epoch < MAX_EPOCH; epoch++) {
		shuffle(begin(train_ids), end(train_ids), rng);

		float train_loss = 0;
		for (unsigned ofs = 0; ofs < num_train_sents; ofs += minibatch) {
			const vector<unsigned> batch_ids(
					begin(train_ids) + ofs,
					begin(train_ids) + min<unsigned>(
						ofs+minibatch, num_train_sents));
			const auto batch = make_batch(train_corpus, batch_ids, eos_id);

			g.clear();

			const auto loss = rnnlm.loss(batch);
			train_loss += loss.to_float() * batch_ids.size();

			optimizer.reset_gradients();
			loss.backward();
			optimizer.update();

			cerr << "\r" << ofs << "/" << num_train_sents << flush;
		}
		cerr << endl;
		const float train_ppl = exp(train_loss / num_train_labels);

		float test_loss = 0;
		for (unsigned ofs = 0; ofs < num_test_sents; ofs += minibatch) {
			const vector<unsigned> batch_ids(
					begin(test_ids) + ofs,
					begin(test_ids) + min<unsigned>(
						ofs+minibatch, num_test_sents));
			const auto batch = make_batch(test_corpus, batch_ids, eos_id);

			g.clear();

			const auto loss = rnnlm.loss(batch);
			test_loss += loss.to_float() * batch_ids.size();

			cerr << "\r" << ofs << "/" << num_test_sents << flush;
		}
		cerr << endl;
		const float test_ppl = exp(test_loss / num_test_labels);

		cerr << "epoch = " << epoch + 1 << "/" << MAX_EPOCH << ", ";
		cerr << "train ppl = " << train_ppl << ", ";
		cerr << "test ppl = " << test_ppl << ", ";
		cerr << endl;
	}

	return 0;
}
