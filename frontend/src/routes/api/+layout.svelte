<script>
	const navItems = [
		{ label: 'METAR', path: '/api/metar', description: 'Current airport weather observations' },
		{ label: 'TAF', path: '/api/taf', description: 'Terminal aerodrome forecasts' },
		{ label: 'PIREP', path: '/api/pirep', description: 'Pilot weather reports' },
		{ label: 'SIGMET/AIRMET', path: '/api/sigmet', description: 'Significant meteorological advisories' },
		{ label: 'Documentation', path: '/api/docs', description: 'API documentation and guides' }
	];
	
	// Get the current path to highlight the active link
	let currentPath = '';
	
	if (typeof window !== 'undefined') {
		currentPath = window.location.pathname;
	}
</script>

<div class="min-h-screen bg-gray-100">
	<header class="bg-white shadow">
		<nav class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
			<div class="flex justify-between h-16">
				<div class="flex">
					<div class="flex-shrink-0 flex items-center">
						<a href="/" class="text-blue-600 font-bold text-lg">Aviation Weather API Hub</a>
					</div>
					
					<div class="hidden sm:ml-6 sm:flex sm:space-x-8">
						{#each navItems as item}
							<a
								href={item.path}
								class={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
									currentPath === item.path || (currentPath.startsWith(item.path) && item.path !== '/')
										? 'border-blue-500 text-gray-900'
										: 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
								}`}
							>
								{item.label}
							</a>
						{/each}
					</div>
				</div>
				
				<div class="hidden sm:ml-6 sm:flex sm:items-center">
					<a
						href="/api"
						class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
					>
						API Dashboard
					</a>
				</div>
				
				<div class="-mr-2 flex items-center sm:hidden">
					<!-- Mobile menu button -->
					<button
						type="button"
						class="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
					>
						<span class="sr-only">Open main menu</span>
						<svg class="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
						</svg>
					</button>
				</div>
			</div>
		</nav>
	</header>
	
	<main>
		<slot />
	</main>
	
	<footer class="bg-white mt-12">
		<div class="max-w-7xl mx-auto py-6 px-4 sm:px-6 md:flex md:items-center md:justify-between lg:px-8">
			<div class="flex justify-center space-x-6 md:order-2">
				<a href="https://github.com" class="text-gray-400 hover:text-gray-500">
					<span class="sr-only">GitHub</span>
					<svg class="h-6 w-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
						<path fill-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clip-rule="evenodd" />
					</svg>
				</a>
			</div>
			<div class="mt-8 md:mt-0 md:order-1">
				<p class="text-center text-base text-gray-400">
					&copy; 2025 Aviation Weather API Hub. All rights reserved.
				</p>
			</div>
		</div>
	</footer>
</div>