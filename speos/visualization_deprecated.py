import numpy as np
import matplotlib.pyplot as plt


class UMAPWrapper:
    def __init__(self, n_components=2, init='random', random_state=0, **kwargs):
        import umap

        self.n_components = n_components
        self.init = init
        self.random_state = 0
        self.umap = umap.UMAP(n_components=n_components, init=init, random_state=random_state, **kwargs)
        self.transformed_embeddings = None

    def fit_transform(self, embeddings):
        self.transformed_embeddings = self.umap.fit_transform(embeddings)
        return self.transformed_embeddings

    def transform(self, embeddings):
        transformed_embeddings = self.umap.transform(embeddings)
        return transformed_embeddings

    def fit(self, embeddings):
        self.embeddings = embeddings
        self.trained_mapper = self.umap.fit(embeddings)

    def plot(self, title="umap.png", node_names=None, color_labels=None):
        plt.figure()
        plt.clf()
        plt.title('Umap of trained Embeddings')
        plt.xlabel('Dim 0')
        plt.ylabel('Dim 1')
        plt.scatter(self.transformed_embeddings[:, 0], self.transformed_embeddings[:, 1])
        if color_labels is not None:
            plt.scatter(self.transformed_embeddings[:, 0], self.transformed_embeddings[:, 1], color=color_labels)
        if node_names is not None:
            for i, txt in enumerate(node_names):
                plt.annotate(txt, (self.transformed_embeddings[i, 0], self.transformed_embeddings[i, 1]), fontsize=6)
        plt.savefig(title)

    def render(self, embeddings=None, node_names=None, color_labels=None, title=None):
        fig = plt.figure()
        plt.clf()
        if title is None:
            plt.title('Umap of trained Embeddings')
        else:
            plt.title('Umap of trained Embeddings\n{}'.format(title))
        plt.xlabel('Dim 0')
        plt.ylabel('Dim 1')
        fig.tight_layout(pad=0)
        if embeddings is None:
            embeddings = self.transformed_embeddings

        if color_labels is not None:
            plt.scatter(embeddings[:, 0], embeddings[:, 1], c=color_labels)
        else:
            plt.scatter(embeddings[:, 0], embeddings[:, 1])
        if node_names is not None:
            for i, txt in enumerate(node_names):
                plt.annotate(txt, (embeddings[i, 0], embeddings[i, 1]), fontsize=6)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape((h, w, 3))
        data = np.transpose(data, (2, 0, 1))
        plt.close()
        return data
