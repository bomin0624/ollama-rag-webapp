<script lang="ts">
	import { browser } from '$app/environment';
	import { queryRag } from '$lib/api/rag';
	import type { ChatEntry, ChatSession } from '$lib/types/chat';
	import { onMount } from 'svelte';

	const storageKey = 'rag-chat-history';

	let chats = $state<ChatSession[]>([]);
	let activeChatId = $state<string | null>(null);
	let query = $state('');
	let error = $state('');
	let isLoading = $state(false);
	let hasLoaded = $state(false);
	let chatPendingDeletion = $state<string | null>(null);
	let activeChat = $derived(chats.find((chat) => chat.id === activeChatId) ?? null);

	onMount(() => {
		const savedChats = localStorage.getItem(storageKey);

		if (savedChats) {
			try {
				const parsedChats = JSON.parse(savedChats) as ChatSession[];
				chats = Array.isArray(parsedChats) ? parsedChats : [];
			} catch {
				chats = [];
			}
		}

		if (chats.length === 0) {
			const firstChat = createChatSession();
			chats = [firstChat];
			activeChatId = firstChat.id;
		} else {
			activeChatId = chats[0].id;
		}

		hasLoaded = true;
	});

	$effect(() => {
		if (browser && hasLoaded) {
			localStorage.setItem(storageKey, JSON.stringify(chats));
		}
	});

	function createId() {
		return crypto.randomUUID();
	}

	function createChatSession(): ChatSession {
		return {
			id: createId(),
			title: 'New chat',
			entries: [],
			updatedAt: Date.now()
		};
	}

	function createChat() {
		const newChat = createChatSession();
		chats = [newChat, ...chats];
		activeChatId = newChat.id;
		query = '';
		error = '';
	}

	function selectChat(chatId: string) {
		activeChatId = chatId;
		error = '';
	}

	function requestDeleteChat(event: MouseEvent, chatId: string) {
		event.stopPropagation();
		chatPendingDeletion = chatId;
	}

	function deleteChat() {
		const chatId = chatPendingDeletion;
		if (!chatId) return;

		chatPendingDeletion = null;

		const remainingChats = chats.filter((chat) => chat.id !== chatId);
		chats = remainingChats;

		if (activeChatId === chatId) {
			if (remainingChats.length > 0) {
				activeChatId = remainingChats[0].id;
			} else {
				createChat();
			}
		}
	}

	function makeTitle(question: string) {
		return question.length > 28 ? `${question.slice(0, 28)}...` : question;
	}

	async function submitQuery() {
		const question = query.trim();

		if (!question || isLoading) return;

		let chatId = activeChatId;
		if (!chatId) {
			createChat();
			chatId = activeChatId;
		}

		if (!chatId) return;

		isLoading = true;
		error = '';

		try {
			const response = await queryRag(question);
			const entry: ChatEntry = { id: createId(), query: question, response };

			chats = chats
				.map((chat) => {
					if (chat.id !== chatId) return chat;

					return {
						...chat,
						title: chat.entries.length === 0 ? makeTitle(question) : chat.title,
						entries: [...chat.entries, entry],
						updatedAt: Date.now()
					};
				})
				.sort((first, second) => second.updatedAt - first.updatedAt);
			query = '';
		} catch (caughtError) {
			error =
				caughtError instanceof Error
					? caughtError.message
					: 'An unexpected error occurred. Please try again.';
		} finally {
			isLoading = false;
		}
	}
</script>

<svelte:head>
	<title>RAG Terminal</title>
	<meta name="description" content="A terminal-style interface for a local RAG knowledge base." />
</svelte:head>

<main class="min-h-screen bg-[#060707] font-mono text-slate-200">
	<div class="mx-auto grid min-h-screen max-w-[1600px] lg:grid-cols-[240px_minmax(0,1fr)]">
		<aside class="hidden border-r border-white/10 bg-black/25 lg:flex lg:flex-col">
			<div class="border-b border-white/10 px-5 py-7">
				<p class="text-xl font-bold tracking-[0.18em] text-cyan-300">RAG</p>
				<p class="mt-2 text-[10px] tracking-[0.15em] text-slate-600">V0.1 // LOCAL</p>
			</div>

			<div class="p-4">
				<button
					onclick={createChat}
					class="w-full border border-cyan-300 bg-cyan-300 px-3 py-2 text-left text-xs font-bold tracking-wider text-black transition hover:bg-cyan-100"
				>
					+ NEW CHAT
				</button>
			</div>

			<nav class="min-h-0 flex-1 overflow-y-auto px-3 pb-4" aria-label="Chat history">
				<p class="mb-2 px-2 text-[10px] font-bold tracking-[0.16em] text-slate-600">CHAT HISTORY</p>
				<div class="space-y-1">
					{#each chats as chat (chat.id)}
						<div
							class={`group flex w-full border-l-2 transition hover:bg-white/5 ${
								chat.id === activeChatId
									? 'border-cyan-300 bg-cyan-300/5 text-cyan-200'
									: 'border-transparent text-slate-500'
							}`}
						>
							<button
								onclick={() => selectChat(chat.id)}
								class="min-w-0 flex-1 px-3 py-3 text-left text-sm"
							>
								<span class="block truncate">{chat.title}</span>
								<span class="mt-1 block text-[10px] text-slate-700"
									>{chat.entries.length} QUERIES</span
								>
							</button>
							<button
								onclick={(event) => requestDeleteChat(event, chat.id)}
								class="m-2 grid size-7 place-items-center text-slate-700 opacity-0 transition group-hover:opacity-100 hover:bg-red-500/10 hover:text-red-400 focus:opacity-100"
								aria-label={`Delete ${chat.title}`}
								title="Delete chat"
							>
								<svg
									viewBox="0 0 24 24"
									fill="none"
									stroke="currentColor"
									stroke-width="1.8"
									class="size-4"
									aria-hidden="true"
								>
									<path d="M3 6h18M8 6V4h8v2m-9 0 1 14h8l1-14M10 10v6m4-6v6" />
								</svg>
							</button>
						</div>
					{/each}
				</div>
			</nav>

			<div class="border-t border-white/10 px-5 py-5">
				<p class="text-xs font-bold text-slate-400">LOCAL_STORAGE</p>
				<p class="mt-1 text-xs text-emerald-400">STATUS: SAVED</p>
			</div>
		</aside>

		<div class="min-w-0">
			<header
				class="flex min-h-18 items-center justify-between gap-4 border-b border-white/10 px-5 py-4 sm:px-8"
			>
				<p class="truncate text-xs font-bold tracking-wide text-slate-500 sm:text-sm">
					ROOT <span class="text-slate-700">/</span> RAG <span class="text-slate-700">/</span>
					<span class="text-slate-200">{activeChat?.title ?? 'NEW CHAT'}</span>
				</p>
				<div class="flex shrink-0 items-center gap-2 text-[10px] font-bold sm:text-xs">
					<button
						onclick={createChat}
						class="border border-cyan-400/70 px-2 py-1 text-cyan-300 lg:hidden"
					>
						+ NEW
					</button>
					<span class="border border-emerald-500/70 px-2 py-1 text-emerald-400">API: LOCAL</span>
				</div>
			</header>

			<div class="mx-auto max-w-6xl p-5 sm:p-8">
				<section id="query" class="border border-white/15 bg-black/20 p-4 sm:p-5">
					<form
						class="border border-white/15 bg-[#080b0b]"
						onsubmit={(event) => {
							event.preventDefault();
							submitQuery();
						}}
					>
						<label class="sr-only" for="query-input">Question</label>
						<div class="flex items-start gap-3 px-4 py-4">
							<span class="pt-1 text-sm font-bold whitespace-nowrap text-cyan-300"
								>rag@local:~$</span
							>
							<textarea
								id="query-input"
								bind:value={query}
								placeholder="type your query..."
								rows="2"
								maxlength="1000"
								class="min-h-14 flex-1 resize-y bg-transparent text-sm leading-6 text-slate-100 outline-none placeholder:text-slate-600"
							></textarea>
						</div>
						<div class="flex items-center justify-between border-t border-white/10 px-4 py-3">
							<p class="text-[10px] tracking-wider text-slate-600">MAX INPUT: 1000 CHARACTERS</p>
							<button
								type="submit"
								disabled={isLoading || !query.trim()}
								class="bg-cyan-300 px-4 py-2 text-xs font-bold tracking-wider text-black transition hover:bg-cyan-100 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-600"
							>
								{isLoading ? 'PROCESSING...' : 'RUN QUERY'}
							</button>
						</div>
					</form>
				</section>

				{#if isLoading}
					<p class="mt-5 text-sm text-cyan-300" aria-live="polite">
						&gt;&gt; Retrieving source documents and generating response...
					</p>
				{/if}

				{#if error}
					<div
						class="mt-5 border border-red-400/50 bg-red-950/20 p-4 text-sm text-red-200"
						role="alert"
					>
						<span class="font-bold text-red-400">ERROR:</span>
						{error}
					</div>
				{/if}

				{#if activeChat?.entries.length}
					<section class="mt-5 space-y-4" aria-label="Chat messages">
						{#each activeChat.entries as entry (entry.id)}
							<article class="border border-white/15 bg-black/20">
								<div class="border-b border-white/10 px-5 py-4 text-sm text-slate-100">
									<span class="font-bold text-cyan-300">rag@local:~$</span>
									<span class="ml-3 whitespace-pre-wrap">{entry.query}</span>
								</div>
								<div
									class="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-3 text-xs"
								>
									<h2 class="font-bold tracking-widest text-cyan-300">RESPONSE_OUTPUT</h2>
									<span class="text-slate-500">MODEL: {entry.response.model}</span>
								</div>
								<p class="px-5 py-5 text-sm leading-7 whitespace-pre-wrap text-slate-200">
									{entry.response.answer}
								</p>

								<div class="border-t border-white/10 px-5 py-4">
									<h3 class="mb-3 text-xs font-bold tracking-widest text-cyan-300">
										SOURCE_DOCUMENTS [{entry.response.sources.length}]
									</h3>
									<div class="space-y-2">
										{#each entry.response.sources as source (source.id)}
											<details class="border border-white/15 bg-[#080b0b]">
												<summary class="cursor-pointer px-4 py-3 text-sm marker:text-cyan-300">
													<span class="text-cyan-300">&gt;</span>
													<span class="ml-2 text-slate-200"
														>{source.title ?? 'untitled-source'}</span
													>
													<span class="ml-2 text-xs text-slate-600">[{source.id}]</span>
												</summary>
												<p
													class="border-t border-white/10 px-4 py-3 text-sm leading-6 text-slate-400"
												>
													{source.content}
												</p>
											</details>
										{/each}
									</div>
								</div>
							</article>
						{/each}
					</section>
				{/if}
			</div>
		</div>
	</div>

	{#if chatPendingDeletion}
		<div
			class="fixed inset-0 z-50 grid place-items-center bg-black/75 p-4 backdrop-blur-sm"
			role="presentation"
		>
			<dialog
				open
				class="fixed top-1/2 right-auto bottom-auto left-1/2 z-50 m-0 block h-fit w-[calc(100%-2rem)] max-w-md -translate-x-1/2 -translate-y-1/2 border border-cyan-300/60 bg-[#080b0b] shadow-2xl shadow-cyan-950/30"
				aria-labelledby="delete-dialog-title"
			>
				<div class="flex items-center justify-between border-b border-white/10 px-5 py-3">
					<h2 id="delete-dialog-title" class="text-xs font-bold tracking-[0.16em] text-cyan-300">
						CONFIRM_DELETION
					</h2>
					<span class="text-xs text-red-400">[ WARNING ]</span>
				</div>
				<div class="px-5 py-6 text-sm leading-7 text-slate-300">
					<p>
						<span class="font-bold text-cyan-300">&gt;</span> Delete this chat and all of its saved queries?
					</p>
					<p class="mt-2 text-xs text-slate-600">THIS ACTION CANNOT BE UNDONE.</p>
				</div>
				<div
					class="flex justify-end gap-3 border-t border-white/10 px-5 py-4 text-xs font-bold tracking-wider"
				>
					<button
						onclick={() => (chatPendingDeletion = null)}
						class="border border-slate-600 px-3 py-2 text-slate-300 transition hover:border-slate-300 hover:text-white"
					>
						CANCEL
					</button>
					<button
						onclick={deleteChat}
						class="border border-red-400 bg-red-400 px-3 py-2 text-black transition hover:bg-red-300"
					>
						DELETE CHAT
					</button>
				</div>
			</dialog>
		</div>
	{/if}
</main>
