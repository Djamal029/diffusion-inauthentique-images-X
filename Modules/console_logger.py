from rich.console import Console
import traceback


class DisplayMessage:
    """
    Permet d'afficher des messages stylés dans la console :
    - Erreur (rouge, ✖)
    - Avertissement (jaune)
    - Information (vert)
    - Succès (vert, ✔)
    """

    def __init__(self):
        self.console = Console()

    def _to_str(self, message) -> str:
        """Convertit n'importe quel objet en chaîne de caractères."""
        if isinstance(message, str):
            return message
        try:
            return str(message)
        except Exception:
            return repr(message)

    def error(self, message):
        message = self._to_str(message)
        self.console.log(f"[bold red]✖ ERROR[/bold red] : [red]{message}[/red]")

    def warning(self, message):
        message = self._to_str(message)
        self.console.log(
            f"[bold yellow]WARNING[/bold yellow] : [yellow]{message}[/yellow]"
        )

    def info(self, message):
        message = self._to_str(message)
        self.console.log(f"[bold green]INFO[/bold green] : {message}")

    def success(self, message):
        message = self._to_str(message)
        self.console.log(f"[bold green]✔ SUCCESS[/bold green] : {message}")

    def exception(self, e: Exception):
        """Affiche une exception complète avec traceback."""
        self.error(f"{e}\n{traceback.format_exc()}")
