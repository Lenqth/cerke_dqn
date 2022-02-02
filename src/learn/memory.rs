use rand::{thread_rng, Rng};

#[derive(Clone)]
pub struct Experience<S, A> {
    pub current_state: S,
    pub action: A,
    pub next_state: S,
    pub value: f32,
}

pub struct Memory<S, A> {
    memory: Vec<Experience<S, A>>,
    capacity: usize,
}

impl<S, A> Memory<S, A> {
    pub fn new() -> Self {
        Self {
            memory: Vec::new(),
            capacity: 50000,
        }
    }

    pub fn put(&mut self, item: Experience<S, A>) {
        if self.memory.len() < self.capacity {
            self.memory.push(item);
        } else {
            let index = thread_rng().gen_range(0..self.capacity);
            self.memory.insert(index, item);
        }
    }

    pub fn sample(&self) -> &Experience<S, A> {
        let index = thread_rng().gen_range(0..self.memory.len());
        self.memory.get(index).unwrap()
    }
}
