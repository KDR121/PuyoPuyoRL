import os.path

from gym_puyopuyo.rendering.state import Garbage, Pop


class SpriteSheet(object):
    BLOCK_WIDTH = 31

    def __init__(self, filename=None):
        import pyglet  # Needs to be a local import to make the package load without a display.
        if not filename:
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, "plain_skin.png")
        self.sheet = pyglet.image.load(str(filename))
        self.grid = pyglet.image.ImageGrid(self.sheet, 16, 16)

    def get_sprite(self, entity, neighbours):
        if isinstance(entity, Garbage):
            return self.grid[3, 6]
        elif isinstance(entity, Pop):
            return self.grid[5, 6 + 2 * entity.sprite_color + entity.age]
        neighbours = [n == entity for n in neighbours]
        index = neighbours[0] + 2 * neighbours[1] + 4 * neighbours[2] + 8 * neighbours[3]
        return self.grid[15 - entity.sprite_color, index]


class ImageViewer(object):
    def __init__(self, width, height, display=None):
        from pyglet.window import Window
        self.sheet = SpriteSheet()
        self.display = display
        self.window = Window(
            width=width * self.sheet.BLOCK_WIDTH,
            height=height * self.sheet.BLOCK_WIDTH,
            display=self.display
        )
        self.isopen = True
        self.init_blend()

    def init_blend(self):
        from pyglet.gl import (
            GL_BLEND, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, glBlendFunc, glEnable
        )
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def render_state(self, state, x_offset=0, flip=True):
        import pyglet
        darken = pyglet.image.SolidColorImagePattern(color=(0, 0, 0, 128))
        if flip:
            self.begin_flip()
        for i, entity in enumerate(state.entities):
            if entity is None:
                continue
            y, x = divmod(i, state.width)
            neighbours = [None] * 4
            neighbours[1] = state[x, y - 1]
            neighbours[0] = state[x, y + 1]
            neighbours[3] = state[x - 1, y]
            neighbours[2] = state[x + 1, y]
            if state.tsu_rules:
                if y == 0:
                    neighbours = [None] * 4
                elif y == 1:
                    neighbours[1] = None
            sprite = self.sheet.get_sprite(entity, neighbours)
            sprite.blit(
                (x + x_offset) * self.sheet.BLOCK_WIDTH,
                (state.height - 1 - y) * self.sheet.BLOCK_WIDTH,
            )
            if state.tsu_rules and y == 0:
                mask = darken.create_image(self.sheet.BLOCK_WIDTH + 1, self.sheet.BLOCK_WIDTH)
                mask.blit(
                    (x + x_offset) * self.sheet.BLOCK_WIDTH,
                    (state.height - 1 - y) * self.sheet.BLOCK_WIDTH,
                )
        for i, deal in enumerate(state.deals):
            for j, entity in enumerate(deal):
                neighbours = [None] * 4
                # Deals are usually not rendered "sticky"
                # neighbours[2 + j] = deal[1 - j]
                sprite = self.sheet.get_sprite(entity, neighbours)
                sprite.blit(
                    (state.width + 2 + j + x_offset) * self.sheet.BLOCK_WIDTH,
                    (state.height - 1 - 2 * i) * self.sheet.BLOCK_WIDTH,
                )

        if flip:
            self.end_flip()

    def begin_flip(self):
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

    def end_flip(self):
        self.window.flip()

    def save_screenshot(self, filename):
        from pyglet.image import get_buffer_manager
        get_buffer_manager().get_color_buffer().save(filename)

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
