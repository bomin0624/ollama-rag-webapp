export type SourceDocument = {
	id: string;
	title: string | null;
	content: string;
};

export type QueryResponse = {
	answer: string;
	sources: SourceDocument[];
	model: string;
};
