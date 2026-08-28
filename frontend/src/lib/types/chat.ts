import type { QueryResponse } from '$lib/types/rag';

export type ChatEntry = {
	id: string;
	query: string;
	response: QueryResponse;
};

export type ChatSession = {
	id: string;
	title: string;
	entries: ChatEntry[];
	updatedAt: number;
};
