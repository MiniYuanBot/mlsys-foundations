//! Part C: SpecInfer-style token trees. Provided (uses your C1 kernel and C2 rule).
//!
//! The draft model expands a STATIC tree shape level by level, e.g.
//! widths = [3, 2, 1]: the root gets its 3 most likely children, each of those
//! its 2 most likely, each of those 1 more -> 1 + 3 + 6 + 6 = 16 nodes.
//! Each level is one draft forward over the whole tree so far (tree attention).
//! The target then verifies the whole tree in ONE forward pass.

use anyhow::Result;
use minillm::gpu::Gpu;

use crate::session::Session;
use crate::spec_student::RoundStats;
use crate::verify_student::tree_accept;

pub struct Tree {
    pub tokens: Vec<u32>,
    pub parents: Vec<usize>,
}

impl Tree {
    pub fn size_for(widths: &[usize]) -> usize {
        let mut level = 1;
        let mut total = 1;
        for &w in widths {
            level *= w;
            total += level;
        }
        total
    }

    pub fn depth(&self, i: usize) -> usize {
        let mut d = 0;
        let mut v = i;
        while v != 0 {
            v = self.parents[v];
            d += 1;
        }
        d
    }
}

/// Build a draft tree whose root is the last committed token.
pub fn build_tree(gpu: &Gpu, draft: &mut Session, widths: &[usize]) -> Result<Tree> {
    anyhow::ensure!(Tree::size_for(widths) <= 64, "tree too large (max 64 nodes)");
    draft.sync_prefix(gpu)?;
    let mut tree = Tree { tokens: vec![*draft.tokens.last().unwrap()], parents: vec![0] };
    let mut frontier = vec![0usize];
    for &w in widths {
        let logits = draft.feed_tree(gpu, &tree.tokens, &tree.parents)?;
        let top = logits.select(gpu, &frontier)?.topk(gpu, w)?;
        let mut next = Vec::new();
        for (&parent, children) in frontier.iter().zip(&top) {
            for &tok in children {
                tree.tokens.push(tok);
                tree.parents.push(parent);
                next.push(tree.tokens.len() - 1);
            }
        }
        frontier = next;
    }
    Ok(tree)
}

/// One tree round: draft builds the tree, target verifies it in one pass.
pub fn tree_round(gpu: &Gpu, target: &mut Session, draft: &mut Session, widths: &[usize]) -> Result<RoundStats> {
    let tree = build_tree(gpu, draft, widths)?;
    let next = target.feed_tree(gpu, &tree.tokens, &tree.parents)?.argmax(gpu)?;
    let (path, bonus) = tree_accept(&tree.tokens, &tree.parents, &next);
    target.commit_tree_path(gpu, &path)?;
    for &v in &path[1..] {
        target.tokens.push(tree.tokens[v]);
    }
    target.tokens.push(bonus);
    // The draft cache holds tree nodes past the committed prefix: forget them.
    let committed = draft.tokens.len() - 1;
    draft.tokens.clone_from(&target.tokens);
    draft.truncate(committed);
    Ok(RoundStats { proposed: tree.tokens.len() - 1, accepted: path.len() - 1 })
}
