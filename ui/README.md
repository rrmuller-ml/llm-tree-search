# Thought Tree Viewer

A single HTML page ([`index.html`](index.html)) that draws a tree saved in the
`Node.to_dict()` format from [`tree.py`](../tree.py) as a top-down graph.

## Running

The page loads `../example.yaml` by default, which needs an HTTP server
(browsers block `fetch` from `file://`). From the repo root:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/ui/>.

- **Another file:** add `?file=` with a path relative to `ui/`, e.g.
  `http://localhost:8000/ui/?file=../other.yaml`. JSON works too.
- **Without a server:** open `index.html` directly and use the
  **Open YAML / JSON…** button.

The YAML parser ([js-yaml](https://github.com/nodeca/js-yaml)) is loaded from
a CDN, so the page needs internet access.

## What it shows

Each box is one `Node` and shows only that node's own fields:

- a header with an id (`n0`, `n1`, …) and its depth
- `thought` (`null` on the root)
- `proposals`, laid out side by side, each with its `title`, `prob` and
  `value` bars, and either `→ nX` (the node it points to) or `unexplored`

Each line runs from the bottom of a node, under the proposal it belongs to, to
the top of the child node. Each depth of the tree is one row, and child nodes
are placed in the same order as their proposals.

`answer` is not shown in the box; it is shown by the node's color.

| Color | Meaning |
| --- | --- |
| Blue node | Root (the question) |
| Gray node | Regular node |
| Green node | Answer node (`answer: true`) |
| Blue proposal, solid border | Explored: has a child node |
| Orange proposal, dashed border | Unexplored: `node` is null |

## Side panel

The panel at the top right shows the structure of a `Node` and a `Proposal`.
Click a node to see that node's actual values in the same layout, including
its `answer` flag, its `parent` id and the node id each proposal points to.
Click the **schema** link or the empty background to go back.

## Controls

- **− / 100% / +**: zoom out, reset, zoom in
- Scroll to move around the tree
- Light and dark mode follow the system setting

## Input format

The same shape `Node.to_dict()` produces; see [`example.yaml`](../example.yaml):

```yaml
proposals:
- title: What is 12 * 15?          # root: single proposal = the question
  node:
    thought: Split the multiplication into easier parts.
    proposals:
    - title: 'Distribute over 15: 12*10 + 12*5'
      value: 0.8
      prob: 0.6
      node:
        thought: 12*10 = 120 and 12*5 = 60.
        proposals:
        - title: 'Sum: 120 + 60'
          value: 0.95
          prob: 1.0
          node:
            thought: 120 + 60 = 180
            answer: true
    - title: 'Double and halve: 6*30'  # unexplored: no node
      value: 0.7
      prob: 0.4
```
