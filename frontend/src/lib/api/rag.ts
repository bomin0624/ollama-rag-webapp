import { PUBLIC_API_BASE_URL } from '$env/static/public';

import type { QueryResponse } from '$lib/types/rag';

const apiBaseUrl = PUBLIC_API_BASE_URL || 'http://localhost:8000';

export async function queryRag(query: string): Promise<QueryResponse> {
	const response = await fetch(`${apiBaseUrl}/query`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ query })
	});

	if (!response.ok) {
		throw new Error('Request failed. Check that the RAG API is running.');
	}

	return response.json() as Promise<QueryResponse>;
}
